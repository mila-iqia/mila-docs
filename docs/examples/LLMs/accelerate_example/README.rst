.. NOTE: This file is auto-generated from examples/LLMs/accelerate_example/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

HuggingFace Accelerate example
==============================

Prerequisites:

* `examples/frameworks/pytorch_setup <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_
* `examples/distributed/single_gpu <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu>`_
* `examples/distributed/multi_gpu <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/multi_gpu>`_
* `examples/distributed/multi_node <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/multi_node>`_

Other interesting resources:

* `<https://sebarnold.net/dist_blog/>`_
* `<https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide>`_
* `<https://huggingface.co/docs/trl/main/en/grpo_trainer#grpo-at-scale-train-a-70b-model-on-multiple-nodes>`_


Click here to see `the code for this example
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/LLMs/accelerate_example>`_

**job.sh**

.. code:: bash

   #!/bin/bash
   #SBATCH --ntasks=2
   #SBATCH --ntasks-per-node=1
   #SBATCH --gpus-per-task=1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem-per-gpu=16GB

   export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
   # Get a unique port for this job based on the job ID
   export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))

   # TODO: Make this work in a multi-node setting with code checkpointing
   # (clone repo to $SLURM_TMPDIR and run things from there), without --nodes=1 so it runs on each node.
   srun --nodes=1 --ntasks-per-node=1 uv sync --offline

   # Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable
   srun uv run --offline bash -c "accelerate launch \
       --machine_rank \$SLURM_NODEID \
       --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
       --num_machines  $SLURM_NNODES --num_processes $SLURM_NTASKS \
       main.py $@"

**main.py**

.. code:: python

   """HuggingFace Example from https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py

   Differences with the reference example:
   - Uses the slurm job ID

   """

   # Copyright 2021 The HuggingFace Inc. team. All rights reserved.
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.
   import argparse
   import dataclasses
   import os
   from pathlib import Path
   import pickle
   import shutil
   import sys
   from typing import Literal
   import tqdm
   from transformers.tokenization_utils_base import BatchEncoding
   from transformers.modeling_utils import PreTrainedModel
   import evaluate
   import simple_parsing
   import torch
   from accelerate import (
       Accelerator,
       DataLoaderConfiguration,
       DistributedType,
   )
   from datasets.dataset_dict import DatasetDict
   from accelerate.utils import set_seed, ProjectConfiguration, InitProcessGroupKwargs
   from datasets import load_dataset
   from torch.optim import AdamW
   from torch.utils.data import DataLoader
   from transformers import (
       AutoModelForSequenceClassification,
       AutoTokenizer,
       PretrainedBartModel,
   )
   from transformers.optimization import get_linear_schedule_with_warmup
   from accelerate.logging import get_logger
   ########################################################################
   # This is a fully working simple example to use Accelerate,
   # specifically showcasing the checkpointing capability,
   # and builds off the `nlp_example.py` script.
   #
   # This example trains a Bert base model on GLUE MRPC
   # in any of the following settings (with the same script):
   #   - single CPU or single GPU
   #   - multi GPUS (using PyTorch distributed mode)
   #   - (multi) TPUs
   #   - fp16 (mixed-precision) or fp32 (normal precision)
   #
   # To help focus on the differences in the code, building `DataLoaders`
   # was refactored into its own function.
   # New additions from the base script can be found quickly by
   # looking for the # New Code # tags
   #
   # To run it in each of these various modes, follow the instructions
   # in the readme for examples:
   # https://github.com/huggingface/accelerate/tree/main/examples
   #
   ########################################################################

   MAX_GPU_BATCH_SIZE = 16
   EVAL_BATCH_SIZE = 32
   logger = get_logger(__name__)

   PREVIOUS_JOB_ID: int | None = None
   if _slurm_job_dependency := os.environ.get("SLURM_JOB_DEPENDENCY"):
       assert _slurm_job_dependency.startswith("afterok:"), _slurm_job_dependency
       job_or_jobs: list[int] = list(
           map(int, _slurm_job_dependency.removeprefix("afterok:").split(":"))
       )
       # IDEA: Do something with this, for instance, load the dataset or checkpoints from the previous job.
       # Currently, since we're not changing anything about the dataset preparation, it gets cached in the HF cache,
       # so there's little need for this atm.


   @dataclasses.dataclass
   class Args:
       mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no"
       """"Whether to use mixed precision.

       Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU.
       """

       cpu: bool = False
       """If passed, will train on the CPU."""

       checkpointing_steps: Literal["epoch"] | str | None = "epoch"
       """Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."""

       output_dir: Path = (
           (
               Path(os.environ["SCRATCH"]) / str(PREVIOUS_JOB_ID)
               if PREVIOUS_JOB_ID
               else os.environ["SLURM_JOB_ID"]
           )
           if "SLURM_JOB_ID" in os.environ
           else Path("./checkpoints")
       )
       """Optional save directory where all checkpoint folders will be stored."""

       resume_from_checkpoint: str | None = None
       """If the training should continue from a checkpoint folder."""

       use_stateful_dataloader: bool = False
       """Whether the dataloader should be a resumable stateful dataloader."""

       lr: float = 2e-5
       """ Learning rate for the optimizer."""

       num_epochs: int = 3
       """ Number of epochs to train for in total."""

       seed: int = 42
       """ Random seed for initialization and reproducibility."""

       batch_size: int = 16
       """Batch size for training."""

       with_tracking: bool = False
       """Whether to load in all available experiment trackers from the environment and use them for logging."""

       only_prepare_dataset: bool = False
       """ When set, return immediately after the dataset is done being prepared, without training.

       This can be useful on SLURM clusters so that a cpu-only job can be used to first prepare the dataset
       before a GPU job is run.
       """


   def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
       """
       Creates a set of `DataLoader`s for the `glue` dataset,
       using "bert-base-cased" as the tokenizer.

       Args:
           accelerator (`Accelerator`):
               An `Accelerator` object
           batch_size (`int`, *optional*):
               The batch size for the train and validation DataLoaders.
       """
       tokenizer_name = "bert-base-cased"
       dataset_name = "glue"
       dataset_task = "mrpc"
       tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
       datasets = load_dataset(dataset_name, dataset_task)

       def tokenize_function(examples):
           # max_length=None => use the model max length (it's actually the default)
           outputs = tokenizer(
               examples["sentence1"],
               examples["sentence2"],
               truncation=True,
               max_length=None,
           )
           return outputs

       # Apply the method we just defined to all the examples in all the splits of the dataset
       # starting with the main process first:

       with accelerator.main_process_first():
           assert isinstance(datasets, DatasetDict)
           tokenized_datasets = datasets.map(
               tokenize_function,
               batched=True,
               remove_columns=["idx", "sentence1", "sentence2"],
               load_from_cache_file=True,
               # cache_file_names={
               #     k: f"{dataset_name}_{dataset_task}_tokenized_{tokenizer_name}_{k}.arrow"
               #     for k in datasets
               # },
               # keep_in_memory=True,
           )
           # tokenized_datasets.save_to_disk()

       # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
       # transformers library
       tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

       def collate_fn(examples):
           # On TPU it's best to pad everything to the same length or training will be very slow.
           max_length = (
               128 if accelerator.distributed_type == DistributedType.XLA else None
           )
           # When using mixed precision we want round multiples of 8/16
           if accelerator.mixed_precision == "fp8":
               pad_to_multiple_of = 16
           elif accelerator.mixed_precision != "no":
               pad_to_multiple_of = 8
           else:
               pad_to_multiple_of = None

           return tokenizer.pad(
               examples,
               padding="longest",
               max_length=max_length,
               pad_to_multiple_of=pad_to_multiple_of,
               return_tensors="pt",
           )

       # Instantiate dataloaders.
       train_dataloader = DataLoader(
           tokenized_datasets["train"],
           shuffle=True,
           collate_fn=collate_fn,
           batch_size=batch_size,
       )
       eval_dataloader = DataLoader(
           tokenized_datasets["validation"],
           shuffle=False,
           collate_fn=collate_fn,
           batch_size=EVAL_BATCH_SIZE,
       )

       return train_dataloader, eval_dataloader


   # For testing only
   if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
       from accelerate.test_utils.training import mocked_dataloaders

       get_dataloaders = mocked_dataloaders  # noqa: F811


   def training_function(args: Args):
       config = args
       # For testing only
       if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
           config = dataclasses.replace(config, num_epochs=2)
       args = dataclasses.replace(args, output_dir=args.output_dir.resolve())
       # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
       lr = config.lr
       num_epochs = config.num_epochs
       seed = config.seed
       batch_size = config.batch_size

       # Initialize accelerator
       dataloader_config = DataLoaderConfiguration(
           use_stateful_dataloader=args.use_stateful_dataloader
       )

       checkpoint_dir: Path | None = max(
           [
               f
               for f in args.output_dir.glob("*_*")
               if f.is_dir() and not f.name.endswith(".tmp")
           ],
           key=lambda f: int(f.stem.rpartition("_")[2]),
           default=None,
       )

       # If the batch size is too big for the GPU, we can use gradient accumulation
       gradient_accumulation_steps = 1
       if batch_size > MAX_GPU_BATCH_SIZE:
           gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
           batch_size = MAX_GPU_BATCH_SIZE

       # kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800),
       #                                 backend="nccl")
       accelerator = Accelerator(
           cpu=args.cpu,
           mixed_precision=args.mixed_precision,
           dataloader_config=dataloader_config,
           gradient_accumulation_steps=gradient_accumulation_steps,
           project_dir=str(args.output_dir.resolve()),
       )

       # New Code #
       # Parse out whether we are saving every epoch or after a certain number of batches
       checkpointing_steps: Literal["epoch"] | int | None
       if args.checkpointing_steps == "epoch":
           checkpointing_steps = args.checkpointing_steps
       elif isinstance(args.checkpointing_steps, str):
           checkpointing_steps = int(args.checkpointing_steps)
       elif args.checkpointing_steps is None:
           # No checkpointing.
           checkpointing_steps = None
       else:
           raise ValueError(
               f"Argument `checkpointing_steps` must be either a number or `epoch`, not `{args.checkpointing_steps}`"
           )

       set_seed(seed)

       train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
       if args.only_prepare_dataset:
           accelerator.print(
               f"Done preparing the dataset, exiting without training (since {args.only_prepare_dataset=})"
           )
           return

       metric = evaluate.load("glue", "mrpc")

       # Instantiate the model (we build the model here so that the seed also control new weights initialization)
       model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
           "bert-base-cased", return_dict=True
       )

       # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
       # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
       # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
       model = model.to(accelerator.device)

       # Instantiate optimizer
       optimizer = AdamW(params=model.parameters(), lr=lr)

       # Instantiate scheduler
       lr_scheduler = get_linear_schedule_with_warmup(
           optimizer=optimizer,
           num_warmup_steps=100,
           num_training_steps=(len(train_dataloader) * num_epochs)
           // gradient_accumulation_steps,
       )

       # Prepare everything
       # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
       # prepare method.
       model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = prepare(
           accelerator, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
       )

       # New Code #
       # We need to keep track of how many total steps we have iterated over
       overall_step = 0
       # We also need to keep track of the stating epoch so files are named properly
       starting_epoch = 0

       skip_first_batches: int | None = None

       def _get_checkpoint_dir(step: int | None = None, epoch: int | None = None):
           assert (step is not None) ^ (epoch is not None), "Use either `step` or `epoch`."
           return args.output_dir / (
               f"step_{step}" if step is not None else f"epoch_{epoch}"
           )

       if checkpoint_dir:
           _int_in_filename = int(checkpoint_dir.stem.rpartition("_")[2])
           if args.checkpointing_steps == "epoch":
               # epoch_0 --> NO training done (initial weights).
               # epoch_1 --> training done for 1 epoch.
               starting_epoch = _int_in_filename
               overall_step = starting_epoch * len(train_dataloader)
               print(f"Resuming training at epoch {starting_epoch} from {checkpoint_dir}")
           else:
               # step_0 --> NO training done (initial weights).
               # step_1 --> 1 training step done.
               overall_step = _int_in_filename
               starting_epoch = overall_step // len(train_dataloader)
               if not args.use_stateful_dataloader:
                   skip_first_batches = overall_step % len(train_dataloader)
               print(f"Resuming training at step {overall_step} in {checkpoint_dir}.")

           # We need to load the checkpoint back in before training here with `load_state`
           # The total number of epochs is adjusted based on where the state is being loaded from,
           # as we assume continuation of the same training script
           accelerator.load_state(input_dir=str(checkpoint_dir))
       elif checkpointing_steps is not None:
           # Save the initial state.
           # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
           # These are saved to folders named `step_{overall_step}` or `epoch_{epoch}` depending on
           # `args.checkpoint_steps`.
           # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and
           # "random_states.pkl"
           # If mixed precision was used, will also save a "scalar.bin" file
           checkpoint_dir = (
               _get_checkpoint_dir(epoch=0)
               if checkpointing_steps == "epoch"
               else _get_checkpoint_dir(step=0)
           )
           save_state(accelerator, checkpoint_dir)
           print(f"Saved initial state in {checkpoint_dir}")

       # Now we train the model
       for epoch in tqdm.tqdm(
           range(starting_epoch, num_epochs),
           desc="Training",
           unit="Epochs",
           position=0,
           disable=not (sys.stdout.isatty() and accelerator.is_main_process),
       ):
           model.train()
           # New Code #
           epoch_start_step = 0
           if epoch == starting_epoch and skip_first_batches:
               # We need to skip steps until we reach the resumed step only if we are not using a stateful dataloader
               assert not args.use_stateful_dataloader
               logger.info(f"Skipping first {skip_first_batches} batches")
               active_dataloader = accelerator.skip_first_batches(
                   train_dataloader, skip_first_batches
               )
               epoch_start_step = skip_first_batches
           else:
               # After the first iteration though, we need to go back to the original dataloader
               active_dataloader = train_dataloader

           for batch_index, batch in enumerate(
               tqdm.tqdm(
                   active_dataloader,
                   desc=f"Train epoch {epoch}",
                   unit="samples",
                   unit_scale=batch_size,  # to see samples/s in pbar
                   position=1,
                   disable=not (sys.stdout.isatty() and accelerator.is_main_process),
               ),
               start=epoch_start_step,
           ):
               assert isinstance(batch, BatchEncoding)

               # We could avoid this line since we set the accelerator with `device_placement=True`.
               batch = batch.to(accelerator.device)
               with accelerator.accumulate(model):
                   outputs = model(**batch)
                   loss = outputs.loss
                   loss = loss / gradient_accumulation_steps
                   accelerator.backward(loss)
                   optimizer.step()
                   lr_scheduler.step()
                   optimizer.zero_grad()
               overall_step += 1
               # New Code #
               # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
               # These are saved to folders named `step_{overall_step}`
               # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
               # If mixed precision was used, will also save a "scalar.bin" file
               if (
                   isinstance(checkpointing_steps, int)
                   and overall_step % checkpointing_steps == 0
               ):
                   checkpoint_dir = _get_checkpoint_dir(step=overall_step)
                   save_state(accelerator, checkpoint_dir)
                   logger.info(f"Saved checkpoint in {checkpoint_dir}")
           model.eval()
           for batch_index, batch in enumerate(eval_dataloader):
               assert isinstance(batch, BatchEncoding)
               # We could avoid this line since we set the accelerator with `device_placement=True` (the default).
               batch = batch.to(accelerator.device)
               with torch.no_grad():
                   outputs = model(**batch)
               predictions = outputs.logits.argmax(dim=-1)
               predictions, references = accelerator.gather_for_metrics(
                   (predictions, batch["labels"])
               )
               metric.add_batch(
                   predictions=predictions,
                   references=references,
               )
           eval_metric = metric.compute()
           assert eval_metric is not None
           # Use accelerator.print to print only on the main process.
           accelerator.print(f"epoch {epoch}:", eval_metric)
           accelerator.log(eval_metric, step=overall_step)
           # New Code #
           # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
           # These are saved to folders named `epoch_{epoch}`
           # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
           # If mixed precision was used, will also save a "scalar.bin" file
           if checkpointing_steps == "epoch":
               # Need to increment epoch here, since "epoch_1" means one epoch is done.
               checkpoint_dir = _get_checkpoint_dir(epoch=epoch + 1)
               assert not checkpoint_dir.exists()
               save_state(accelerator, checkpoint_dir)

       # Need to save a new epoch:
       if isinstance(checkpointing_steps, int) and overall_step % checkpointing_steps == 0:
           checkpoint_dir = _get_checkpoint_dir(step=overall_step)
           assert not checkpoint_dir.exists()
           save_state(accelerator, checkpoint_dir)
           logger.info(f"Saved final checkpoint in {checkpoint_dir}")

       accelerator.end_training()


   def prepare[*Ts](accelerator: Accelerator, *args: *Ts) -> tuple[*Ts]:
       """A wrapper around `accelerator.prepare` that preserves the type of the inputs."""
       return accelerator.prepare(*args)


   def save_state(
       accelerator: Accelerator,
       checkpoint_dir: str | Path,
   ):
       """Small convenience wrapper around `accelerator.save_state` with some tweaks.

       - Saves the state in a temporary directory with the suffix `.tmp`, and renames at the end.
         (This is useful to avoid issues when the program is interrupted while saving a checkpoint).

       """
       if not accelerator.is_main_process:
           return
       checkpoint_dir = Path(checkpoint_dir)
       if checkpoint_dir.exists():
           raise RuntimeError(f"Checkpoint directory {checkpoint_dir} already exists!")
       temp_checkpoint_dir = checkpoint_dir.with_suffix(".tmp")
       if temp_checkpoint_dir.exists():
           logger.warning(
               f"Temporary checkpoint directory {checkpoint_dir} already exists (from previous attempt at checkpointing)."
           )
           shutil.rmtree(temp_checkpoint_dir)
       # TODO: Can't actually do this .tmp and rename, because `save_state` apparently does something
       # asynchronously in a subprocess, and by the time it writes, the parent directory doesn't exist
       # anymore, resulting in an error.
       temp_checkpoint_dir = checkpoint_dir
       temp_checkpoint_dir.mkdir(parents=True, exist_ok=False)
       accelerator.save_state(str(temp_checkpoint_dir))
       temp_checkpoint_dir.rename(checkpoint_dir)
       logger.info(f"Saved state in {checkpoint_dir}")


   def parse_args() -> Args:
       return simple_parsing.parse(Args)


   def _parse_args():
       parser = argparse.ArgumentParser(description="Simple example of training script.")
       parser.add_argument(
           "--mixed_precision",
           type=str,
           default=None,
           choices=["no", "fp16", "bf16", "fp8"],
           help="Whether to use mixed precision. Choose"
           "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
           "and an Nvidia Ampere GPU.",
       )
       parser.add_argument(
           "--cpu", action="store_true", help="If passed, will train on the CPU."
       )
       parser.add_argument(
           "--checkpointing_steps",
           type=str,
           default=None,
           help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
       )
       parser.add_argument(
           "--output_dir",
           type=str,
           default=".",
           help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
       )
       parser.add_argument(
           "--resume_from_checkpoint",
           type=str,
           default=None,
           help="If the training should continue from a checkpoint folder.",
       )
       parser.add_argument(
           "--use_stateful_dataloader",
           action="store_true",
           help="If the dataloader should be a resumable stateful dataloader.",
       )
       return parser.parse_args()


   def main():
       args = parse_args()
       # config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
       training_function(args)


   if __name__ == "__main__":
       main()

**pyproject.toml**

.. code:: toml

   [project]
   name = "accelerate_example"
   version = "0.1.0"
   description = "Add your description here"
   readme = "README.md"
   requires-python = ">=3.12"
   dependencies = [
       "accelerate>=1.7.0",
       "datasets>=3.6.0",
       "evaluate>=0.4.4",
       "scikit-learn>=1.7.0",
       "simple-parsing>=0.1.7",
       "transformers>=4.52.4",
   ]


   [tool.uv]
   python-preference = "system"

   ## From https://docs.astral.sh/uv/reference/settings/#index-strategy:
   ## "Only use results from the first index that returns a match for a given package name."
   ## In other words: only get the package from PyPI if there isn't a version of it in the DRAC wheelhouse.
   # index-strategy = "first-index"

   ## "Search for every package name across all indexes, exhausting the versions from the first index before
   ##  moving on to the next"
   ## In other words: Only get the package from PyPI if the requested version is higher than the version
   ## in the DRAC wheelhouse.
   # index-strategy = "unsafe-first-match"

   ## "Search for every package name across all indexes, preferring the "best" version found.
   ##  If a package version is in multiple indexes, only look at the entry for the first index."
   ## In other words: Consider all versions of the package DRAC + PyPI, and use the version that best matches
   ## the requested version. In a tie, choose the DRAC wheel.
   index-strategy = "unsafe-best-match"

   [[tool.uv.index]]
   name = "drac-gentoo2023-x86-64-v3"
   url = "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3"
   format = "flat"

   [[tool.uv.index]]
   name = "drac-gentoo2023-generic"
   url = "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic"
   format = "flat"

   [[tool.uv.index]]
   name = "drac-generic"
   url = "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"
   format = "flat"

**Running this example**

1. Install UV from https://docs.astral.sh/uv

2. On SLURM clusters where you do not have internet access on compute nodes, you need to first create the virtual environment:

.. code-block:: bash

    $ salloc --gpus=1 --cpus-per-task=4 --mem=16G  # Get an interactive job
    $ module load httproxy/1.0  # if on a compute node, use this to get some internet access
    $ uv sync


3. Launch the job:

.. code-block:: bash

    $ sbatch job.sh
