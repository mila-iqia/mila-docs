#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text
file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation

Old bug (fixed with the solution in the last reply to the thread):
- Fix bug: fatal error: cusolverDn.h: No such file or directory
https://github.com/microsoft/DeepSpeed/issues/2684
"""


import os

# TODO: Remove when not running on a SLURM cluster.
SLURM_TMPDIR = os.environ["SLURM_TMPDIR"]
os.environ["HF_HOME"] = SLURM_TMPDIR + "/cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = SLURM_TMPDIR + "/cache/huggingface/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = SLURM_TMPDIR + "/cache/huggingface/hub"

# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import Literal, Optional

import datasets
import rich.logging
import simple_parsing
import torch
import torch.distributed
import transformers
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from accelerate.utils.dataclasses import InitProcessGroupKwargs
from datasets import load_dataset
from huggingface_hub import Repository
from simple_parsing import field
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version

logger = get_logger(__name__)

require_version(
    "datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt"
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Note: These are only used in setting the wandb run name and group, if they are set.
JOB_ID: Optional[str] = os.environ.get("JOB_ID", os.environ.get("SLURM_JOB_ID"))
NODEID: Optional[str] = os.environ.get("NODEID", os.environ.get("SLURM_NODEID"))


@dataclass
class Args:
    output_dir: str
    """Where to store the logs and final model."""

    dataset_name: Optional[str] = "wikitext"
    """The name of the dataset to use (via the datasets library)."""

    dataset_config_name: Optional[str] = "wikitext-103-v1"
    """The configuration name of the dataset to use (via the datasets library)."""

    train_file: Optional[str] = None
    """A csv or a json file containing the training data."""

    validation_file: Optional[str] = None
    """A csv or a json file containing the validation data."""

    validation_split_percentage: int = 5
    """The percentage of the train set used as validation set in case there's no validation
    split."""

    model_name_or_path: Optional[str] = None
    """Path to pretrained model or model identifier from huggingface.co/models."""

    config_name: Optional[str] = None
    """Pretrained config name or path if not the same as model_name."""

    tokenizer_name: Optional[str] = None
    """Pretrained tokenizer name or path if not the same as model_name."""

    use_slow_tokenizer: bool = False
    """If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."""

    per_device_train_batch_size: int = 1
    """Batch size (per device) for the training dataloader."""

    per_device_eval_batch_size: int = 1
    """Batch size (per device) for the evaluation dataloader."""

    learning_rate: float = 5e-5
    """Initial learning rate (after the potential warmup period) to use."""

    weight_decay: float = 0.0
    """Weight decay to use."""

    num_train_epochs: int = 3
    """Total number of training epochs to perform."""

    max_train_steps: Optional[int] = None
    """Total number of training steps to perform.

    If provided, overrides num_train_epochs.
    """

    gradient_accumulation_steps: int = 1
    """Number of updates steps to accumulate before performing a backward/update pass."""

    lr_scheduler_type: SchedulerType = SchedulerType.LINEAR
    """The scheduler type to use."""
    # choices=[
    #     "linear",
    #     "cosine",
    #     "cosine_with_restarts",
    #     "polynomial",
    #     "constant",
    #     "constant_with_warmup",
    # ],

    num_warmup_steps: int = 0
    """Number of steps for the warmup in the lr scheduler."""

    seed: Optional[int] = None
    """A seed for reproducible training."""

    model_type: Optional[str] = field(default=None, choices=MODEL_TYPES)
    """Model type to use if training from scratch."""

    block_size: Optional[int] = None
    """Optional input sequence length after tokenization.

    The training dataset will be truncated in block of this size for training. Default to the model
    max input length for single sentence inputs (take into account special tokens).
    """

    preprocessing_num_workers: Optional[int] = int(os.environ.get("CPUS_PER_GPU", 8))
    """The number of processes to use for the preprocessing."""

    overwrite_cache: bool = False
    """Overwrite the cached training and evaluation sets."""

    no_keep_linebreaks: bool = False
    """Do not keep line breaks when using TXT files."""

    push_to_hub: bool = False
    """Whether or not to push the model to the Hub."""

    hub_model_id: str = ""
    """The name of the repository to keep in sync with the local `output_dir`."""

    hub_token: str = ""
    """The token to use to push to the Model Hub."""

    checkpointing_steps: Optional[str] = None
    """Whether the various states should be saved at the end of every n steps, or 'epoch' for each
    epoch."""

    resume_from_checkpoint: Optional[str] = None
    """If the training should continue from a checkpoint folder."""

    load_best_model: bool = False
    """Whether to load the best model at the end of training."""

    with_tracking: bool = False
    """Whether to enable experiment trackers for logging."""

    report_to: str = "all"
    """The integration to report the results and logs to.

    Supported platforms are `"tensorboard"`, `"wandb"` and `"comet_ml"`. Use `"all"` (default) to
    report to all integrations. Only applicable when `--with_tracking` is passed.
    """

    # ADDED:
    log_every_n_steps: int = 1
    """Logging interval when using trackers (when `--with_tracking` is passed)."""

    wandb_tags: list[str] = field(default_factory=list)

    init_process_group_backend: Literal["nccl", "gloo"] = "nccl"

    def __post_init__(self):
        self.wandb_tags = sum([tag.split(",") for tag in self.wandb_tags], [])

        # Sanity checks
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, json or txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, json or txt file."

        if self.push_to_hub:
            assert (
                self.output_dir is not None
            ), "Need an `output_dir` to create a repo when `--push_to_hub` is passed."


def parse_args() -> Args:
    parser = simple_parsing.ArgumentParser(
        description="Train or Finetune a transformers model on a causal language modeling task"
    )
    # parser = simple_parsing.parse(Args, description="Train or Finetune a transformers model on a causal language modeling task")
    parser.add_arguments(Args, dest="config")
    namespace = parser.parse_args()
    args: Args = namespace.config
    return args


# New Code #
def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries The main purpose for this
    is to be able to resume training from that instant again."""
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


# New Code #
def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries The main purpose for this
    is to be able to resume training from that instant again."""
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict
    return (epoch, last_global_step)


# New Code #
def evaluate(args: Args, model, eval_dataloader, accelerator: Accelerator, eval_dataset):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        eval_loss = float("inf")
        perplexity = float("inf")
    return perplexity, eval_loss


def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    # https://pytorch.org/docs/master/distributed.html#torch.distributed.TCPStore

    @dataclass
    class CustomInitProcessGroupKwargs(InitProcessGroupKwargs):
        # IDEA: `InitProcessGroupKwargs` only has `init_method` and `timeout` entries. I'd add `store` too.
        init_method: Optional[str] = None
        timeout: timedelta = timedelta(seconds=1800)

        # backend: Literal["nccl", "gloo"] = "nccl"
        # store: Optional[Store] = None

        rank: Optional[int] = None
        world_size: Optional[int] = None

    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = os.environ["MASTER_PORT"]

    # assert "RANK" not in os.environ, os.environ["RANK"]
    # os.environ["RANK"] = os.environ["SLURM_PROCID"]
    # RANK = os.environ["RANK"]

    init_process_group_kwargs = CustomInitProcessGroupKwargs(
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        # Reduced the timeout here, so the job fails quicker if there's a communication problem between nodes.
        timeout=timedelta(seconds=120),
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        # backend=args.init_process_group_backend,
    )
    if args.init_process_group_backend != "nccl":
        # NOTE: In `state.py` of Accelerate, line 117, it checks if the process group is already
        # initialized, and if not, it does init_process_group(backend="nccl", **kwargs). Therefore,
        # if we want to change the backend used, we need to initialize the process group ourselves.
        torch.distributed.init_process_group(
            backend=args.init_process_group_backend,
            **init_process_group_kwargs.to_kwargs(),
        )

    accelerator = Accelerator(
        log_with=[args.report_to] if args.with_tracking else None,
        project_dir=args.output_dir,
        kwargs_handlers=[init_process_group_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=True,
    )
    # Make one log on every process with the configuration for debugging.

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{accelerator.process_index}/{accelerator.num_processes}] %(name)s - %(message)s ",
        handlers=[
            rich.logging.RichHandler(markup=True, tracebacks_width=120)
        ],  # Very pretty, uses the `rich` package.
    )
    logger.info(args, main_process_only=True)
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"HF_HOME={os.environ['HF_HOME']}")
    logger.info(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")
    logger.info(f"HUGGINGFACE_HUB_CACHE={os.environ['HUGGINGFACE_HUB_CACHE']}")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        # TODO: Remove, never used:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        # TODO: Only used if dataset is a local (e.g. csv/json) file
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
        )
    elif args.config_name:
        # Use the tokenizer with the same name as the model.
        tokenizer = AutoTokenizer.from_pretrained(
            args.config_name, use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    # NOTE: Use `local_main_process_first` if the dataset is on a node-local filesystem (e.g.
    # SLURM_TMPDIR), `main_process_first` otherwise.
    with accelerator.local_main_process_first():
        logger.info(f"Tokenizing! HF_HOME: {os.environ['HF_HOME']}")
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples: dict, block_size: int):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.local_main_process_first():
        logger.info(f"Grouping! HF_HOME: {os.environ['HF_HOME']}")
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            fn_kwargs={"block_size": block_size},
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            # TODO: See if this makes things faster. (invalidates the cache atm)
            # keep_in_memory=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        # TODO: Load the dataset in memory to speed up training (if dataloading is a bottleneck)
        # fast_local_dir = Path(SLURM_TMPDIR) / "data"
        # lm_datasets.save_to_disk(str(fast_local_dir))
        # from datasets import load_from_disk
        # lm_datasets = load_from_disk(str(fast_local_dir), keep_in_memory=True)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(no_decay_str in n for no_decay_str in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(no_decay_str in n for no_decay_str in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates AdamW
    # Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Scheduler and math around the number of training steps.

    # NOTE: When passing the gradient accumulation steps to the Accelerator constructor, it already
    # takes care of updating the value in the DeepSpeed plugin.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accelerator.gradient_accumulation_steps
    )
    # BUG: (@lebrice): Overwrites `max_train_steps` when `max_train_steps` < 1 epoch!
    # args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if args.max_train_steps < len(train_dataloader):
        # TODO: Unsure how to modify this here when using gradient accumulation: do we keep the
        # number of updates constant? Or the number of "used samples" constant?
        args.max_train_steps = args.max_train_steps
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Figure out how many steps we should save the Accelerator states
    if isinstance(args.checkpointing_steps, str):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_local_main_process and args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        experiment_config["env"] = os.environ.copy()
        wandb.init(
            project="llm_training",
            config=experiment_config,
            name=f"{JOB_ID}_{NODEID}" if JOB_ID is not None else None,
            group=JOB_ID if JOB_ID is not None else None,
            tags=args.wandb_tags,
        )
        # NOTE: IF you want to use tensorboard instead of wandb, use `init_trackers` instead.
        # I (@lebrice) decided to use wandb directly to get the metrics from all nodes.
        # TODO: It seems like the trackers from wandb should actually log from each node, but I
        # don't have time to test it out.
        # accelerator.init_trackers(
        #     "llm_training",
        #     experiment_config,
        #     init_kwargs={
        #         "wandb": dict(
        #             name=f"{JOB_ID}_{NODEID}" if JOB_ID is not None else None,
        #             group=JOB_ID if JOB_ID is not None else None,
        #         ),
        #     },
        # )

    # Train!
    # TODO: Double-check that this is indeed correct, since it impacts our calculation for the throughput.
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None

    resume_step = None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        _, last_global_step = load_training_checkpoint(
            model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        resume_step = last_global_step
        starting_epoch = resume_step // len(train_dataloader)
        resume_step -= starting_epoch * len(train_dataloader)

    # NOTE (@lebrice): Added these for the throughput metrics below.
    start_time: Optional[float] = None
    last_log_time: Optional[float] = None
    n_updates_since_start_of_run: int = 0
    n_updates_since_last_log: int = 0
    throughput_samples_per_sec: float = 0.0  # instantaneous throughput
    throughput_samples_per_sec_since_start: float = 0.0  # Average throughput since start of run

    total_loss = 0.0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                accelerator.backward(loss)
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()

            if accelerator.optimizer_step_was_skipped:
                continue

            completed_steps += 1
            progress_bar.update(1)

            if args.with_tracking:
                total_loss += loss.detach().float()

            if accelerator.sync_gradients:  # True when performing the update.
                # NOTE: Added (@lebrice) for throughput metrics.
                n_updates_since_start_of_run += 1
                n_updates_since_last_log += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if accelerator.is_local_main_process and completed_steps % args.log_every_n_steps == 0:
                if accelerator.optimizer_step_was_skipped:
                    start_time = time.time()
                elif start_time is None:
                    # The first time we get here (first logging call), we've never logged before,
                    # so we can't calculate the throughput. Only save the start time for the next
                    # call.
                    start_time = time.time()
                else:
                    seconds_since_start = time.time() - start_time
                    seconds_since_last_log = time.time() - (last_log_time or start_time)

                    # TODO: Not 100% sure, but seems like we're only logging values on the first node,
                    # so we assume that one update here = one update on all nodes = total_batch_size
                    # samples.
                    n_samples_since_start = n_updates_since_start_of_run * total_batch_size
                    n_samples_since_last_log = n_updates_since_last_log * total_batch_size

                    throughput_samples_per_sec = n_samples_since_last_log / seconds_since_last_log
                    throughput_samples_per_sec_since_start = (
                        n_samples_since_start / seconds_since_start
                    )
                    updates_per_sec = n_updates_since_last_log / seconds_since_last_log
                    updates_per_sec_since_start = (
                        n_updates_since_start_of_run / seconds_since_start
                    )
                    # TODO: If we want to use tensorboard, use `accelerator.log` instead.
                    # accelerator.log(
                    wandb.log(
                        {
                            "train_loss": loss.detach().item(),
                            "samples_per_sec": throughput_samples_per_sec,
                            "secs_per_sample": 1 / throughput_samples_per_sec,
                            "avg_samples_per_sec": throughput_samples_per_sec_since_start,
                            "updates_per_sec": updates_per_sec,
                            "secs_per_update": 1 / updates_per_sec,
                            "avg_updates_per_sec": updates_per_sec_since_start,
                            "epoch": epoch,
                            "step": completed_steps,
                            "n_samples": completed_steps * total_batch_size,
                        },
                        step=completed_steps,
                    )

                last_log_time = time.time()
                n_updates_since_last_log = 0

            if completed_steps >= args.max_train_steps:
                break

        perplexity, eval_loss = evaluate(args, model, eval_dataloader, accelerator, eval_dataset)
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if accelerator.is_local_main_process and args.with_tracking:
            # accelerator.log(
            wandb.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_epoch_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        # Save the DeepSpeed checkpoint to the specified path
        checkpoint_model(args.output_dir, epoch, model, epoch, completed_steps)

        # Tracks the best checkpoint and best metric
        if best_metric is None or best_metric > perplexity:
            best_metric = perplexity
            best_metric_checkpoint = os.path.join(args.output_dir, str(epoch))
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")

    # Loads the best checkpoint after the training is finished
    if args.load_best_model:
        _, last_global_step = load_training_checkpoint(
            model,
            "/".join(best_metric_checkpoint.split("/")[:-1]),
            tag=best_metric_checkpoint.split("/")[-1],
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )

    # Evaluates using the best checkpoint
    perplexity, eval_loss = evaluate(args, model, eval_dataloader, accelerator, eval_dataset)
    logger.info(f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")
    if perplexity != best_metric:
        raise AssertionError(
            f"Best metric {best_metric} does not match the metric {perplexity} of the loaded best model."
        )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        # New Code #
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity, "eval_loss": eval_loss.item()}, f)


if __name__ == "__main__":
    main()
