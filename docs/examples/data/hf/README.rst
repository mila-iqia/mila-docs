Hugging Face Dataset
====================


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/data/hf>`_


**job.sh**

.. code:: diff

    # distributed/001_single_gpu/job.sh -> data/hf/job.sh
    #!/bin/bash
    #SBATCH --gpus-per-task=rtx8000:1
   -#SBATCH --cpus-per-task=4
   +#SBATCH --cpus-per-task=12
    #SBATCH --ntasks-per-node=1
   -#SBATCH --mem=16G
   -#SBATCH --time=00:15:00
   +#SBATCH --mem=48G
   +#SBATCH --time=04:00:00
   +#SBATCH --tmp=1500G
   +set -o errexit
   +
   +function wrap_cmd {
   +    for a in "$@"
   +    do
   +        echo -n "\"$a\" "
   +    done
   +}


    # Echo time and hostname into log
    echo "Date:     $(date)"
    echo "Hostname: $(hostname)"


    # Ensure only anaconda/3 module loaded.
    module --quiet purge
    # This example uses Conda to manage package dependencies.
    # See https://docs.mila.quebec/Userguide.html#conda for more information.
    module load anaconda/3
    module load cuda/11.7

   +
    # Creating the environment for the first time:
    # conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
   -#     pytorch-cuda=11.7 -c pytorch -c nvidia
   +#     pytorch-cuda=11.7 scipy -c pytorch -c nvidia
    # Other conda packages:
   -# conda install -y -n pytorch -c conda-forge rich tqdm
   +# conda install -y -n pytorch -c conda-forge rich tqdm datasets

    # Activate pre-existing environment.
    conda activate pytorch


   -# Stage dataset into $SLURM_TMPDIR
   -mkdir -p $SLURM_TMPDIR/data
   -cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
   -# General-purpose alternatives combining copy and unpack:
   -#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
   -#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/
   +if [[ -z "$HF_DATASETS_CACHE" ]]
   +then
   +    # Store the huggingface datasets cache in $SCRATCH
   +    export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
   +fi
   +if [[ -z "$HUGGINGFACE_HUB_CACHE" ]]
   +then
   +    # Store the huggingface hub cache in $SCRATCH
   +    export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub
   +fi
   +if [[ -z "$_DATA_PREP_WORKERS" ]]
   +then
   +    _DATA_PREP_WORKERS=$SLURM_JOB_CPUS_PER_NODE
   +fi
   +if [[ -z "$_DATA_PREP_WORKERS" ]]
   +then
   +    _DATA_PREP_WORKERS=16
   +fi
   +
   +
   +# Preprocess the dataset and cache the result such that the heavy work is done
   +# only once *ever*
   +# Required conda packages:
   +# conda install -y -c conda-forge zstandard
   +srun --ntasks=1 --ntasks-per-node=1 \
   +    time -p python3 prepare_data.py "/network/datasets/pile" $_DATA_PREP_WORKERS
   +
   +
   +# Copy the preprocessed dataset to $SLURM_TMPDIR so it is close to the GPUs for
   +# faster training. This should be done once per compute node
   +cmd=(
   +    # Having 'bash' here allows the execution of a script file which might not
   +    # have the execution flag on
   +    bash data.sh
   +    # The current dataset cache dir
   +    "$HF_DATASETS_CACHE"
   +    # The local dataset cache dir
   +    # Use '' to lazy expand the expression such as $SLURM_TMPDIR will be
   +    # interpreted on the local compute node rather than the master node
   +    '$SLURM_TMPDIR/data'
   +    $_DATA_PREP_WORKERS
   +)
   +# 'time' will objectively give a measure for the copy of the dataset. This can
   +# be used to compare the timing of multiple code versionw and make sure any slow
   +# down doesn't come from the code itself.
   +srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
   +    time -p bash -c "$(wrap_cmd "${cmd[@]}")"


    # Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
    unset CUDA_VISIBLE_DEVICES

    # Execute Python script
   -python main.py
   +env_var=(
   +    # Use the local copy of the preprocessed dataset
   +    HF_DATASETS_CACHE='"$SLURM_TMPDIR/data"'
   +)
   +cmd=(
   +    python3
   +    main.py
   +)
   +srun bash -c "$(echo "${env_var[@]}") $(wrap_cmd "${cmd[@]}")"


**main.py**

.. code:: diff

    # distributed/001_single_gpu/main.py -> data/hf/main.py
   -"""Single-GPU training example."""
   +"""HuggingFace training example."""
    import logging
   -import os
   -from pathlib import Path

    import rich.logging
    import torch
   -from torch import Tensor, nn
   -from torch.nn import functional as F
   -from torch.utils.data import DataLoader, random_split
   -from torchvision import transforms
   -from torchvision.datasets import CIFAR10
   -from torchvision.models import resnet18
   +from torch import nn
   +from torch.utils.data import DataLoader
    from tqdm import tqdm

   +from py_utils import (
   +    get_dataset_builder, get_num_workers, get_raw_datasets, get_tokenizer,
   +    preprocess_datasets
   +)
   +

    def main():
   -    training_epochs = 10
   -    learning_rate = 5e-4
   -    weight_decay = 1e-4
   -    batch_size = 128
   +    training_epochs = 1
   +    batch_size = 256

        # Check that the GPU is available
        assert torch.cuda.is_available() and torch.cuda.device_count() > 0
        device = torch.device("cuda", 0)

        # Setup logging (optional, but much better than using print statements)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
        )

        logger = logging.getLogger(__name__)

   -    # Create a model and move it to the GPU.
   -    model = resnet18(num_classes=10)
   -    model.to(device=device)
   -
   -    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
   -
   -    # Setup CIFAR10
   +    # Setup ImageNet
        num_workers = get_num_workers()
   -    dataset_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "data"
   -    train_dataset, valid_dataset, test_dataset = make_datasets(str(dataset_path))
   +    train_dataset, valid_dataset, test_dataset = make_datasets(num_workers)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        test_dataloader = DataLoader(  # NOTE: Not used in this example.
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        # Checkout the "checkpointing and preemption" example for more info!
        logger.debug("Starting training from scratch.")

        for epoch in range(training_epochs):
            logger.debug(f"Starting epoch {epoch}/{training_epochs}")

   -        # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
   -        model.train()
   -
            # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Train epoch {epoch}",
            )

            # Training loop
            for batch in train_dataloader:
                # Move the batch to the GPU before we pass it to the model
   -            batch = tuple(item.to(device) for item in batch)
   -            x, y = batch
   -
   -            # Forward pass
   -            logits: Tensor = model(x)
   -
   -            loss = F.cross_entropy(logits, y)
   -
   -            optimizer.zero_grad()
   -            loss.backward()
   -            optimizer.step()
   -
   -            # Calculate some metrics:
   -            n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
   -            n_samples = y.shape[0]
   -            accuracy = n_correct_predictions / n_samples
   +            batch = {k:item.to(device) for k, item in batch.items()}

   -            logger.debug(f"Accuracy: {accuracy.item():.2%}")
   -            logger.debug(f"Average Loss: {loss.item()}")
   +            # [Training of the model goes here]

                # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
                progress_bar.update(1)
   -            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
            progress_bar.close()

   -        val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
   +        val_loss, val_accuracy = validation_loop(None, valid_dataloader, device)
            logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

        print("Done!")


    @torch.no_grad()
    def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
   -    model.eval()
   -
        total_loss = 0.0
        n_samples = 0
        correct_predictions = 0

        for batch in dataloader:
   -        batch = tuple(item.to(device) for item in batch)
   -        x, y = batch
   -
   -        logits: Tensor = model(x)
   -        loss = F.cross_entropy(logits, y)
   +        batch = {k:item.to(device) for k, item in batch.items()}

   -        batch_n_samples = x.shape[0]
   -        batch_correct_predictions = logits.argmax(-1).eq(y).sum()
   +        batch_n_samples = batch["input_ids"].data.shape[0]

   -        total_loss += loss.item()
            n_samples += batch_n_samples
   -        correct_predictions += batch_correct_predictions

        accuracy = correct_predictions / n_samples
        return total_loss, accuracy


   -def make_datasets(
   -    dataset_path: str,
   -    val_split: float = 0.1,
   -    val_split_seed: int = 42,
   -):
   -    """Returns the training, validation, and test splits for CIFAR10.
   -
   -    NOTE: We don't use image transforms here for simplicity.
   -    Having different transformations for train and validation would complicate things a bit.
   -    Later examples will show how to do the train/val/test split properly when using transforms.
   +def make_datasets(num_workers:int=None):
   +    """Returns the training, validation, and test splits for the prepared dataset.
        """
   -    train_dataset = CIFAR10(
   -        root=dataset_path, transform=transforms.ToTensor(), download=True, train=True
   -    )
   -    test_dataset = CIFAR10(
   -        root=dataset_path, transform=transforms.ToTensor(), download=True, train=False
   +    builder = get_dataset_builder()
   +    raw_datasets = get_raw_datasets(builder)
   +    tokenizer = get_tokenizer()
   +    preprocessed_datasets = preprocess_datasets(tokenizer, raw_datasets, num_workers=num_workers)
   +    return (
   +        preprocessed_datasets["train"], preprocessed_datasets["validation"],
   +        preprocessed_datasets["test"]
        )
   -    # Split the training dataset into a training and validation set.
   -    n_samples = len(train_dataset)
   -    n_valid = int(val_split * n_samples)
   -    n_train = n_samples - n_valid
   -    train_dataset, valid_dataset = random_split(
   -        train_dataset, (n_train, n_valid), torch.Generator().manual_seed(val_split_seed)
   -    )
   -    return train_dataset, valid_dataset, test_dataset
   -
   -
   -def get_num_workers() -> int:
   -    """Gets the optimal number of DatLoader workers to use in the current job."""
   -    if "SLURM_CPUS_PER_TASK" in os.environ:
   -        return int(os.environ["SLURM_CPUS_PER_TASK"])
   -    if hasattr(os, "sched_getaffinity"):
   -        return len(os.sched_getaffinity(0))
   -    return torch.multiprocessing.cpu_count()


    if __name__ == "__main__":
        main()


**prepare_data.py**

.. code:: python

   """Preprocess the dataset.
   In this example, HuggingFace is used and the resulting dataset will be stored in
   $HF_DATASETS_CACHE. It is preferable to set the datasets cache to a location in
   $SCRATCH"""

   from py_utils import (
       get_config, get_dataset_builder, get_num_workers, get_raw_datasets,
       get_tokenizer, preprocess_datasets
   )


   if __name__ == "__main__":
       import sys
       import time

       _LOCAL_DS = sys.argv[1]
       try:
           _WORKERS = int(sys.argv[2])
       except IndexError:
           _WORKERS = get_num_workers()

       t = -time.time()
       _ = get_config()
       builder = get_dataset_builder(local_dataset=_LOCAL_DS, num_workers=_WORKERS)
       raw_datasets = get_raw_datasets(builder)
       tokenizer = get_tokenizer()
       _ = preprocess_datasets(tokenizer, raw_datasets, num_workers=_WORKERS)
       t += time.time()

       print(f"Prepared data in {t/60:.2f}m")


**data.sh**

.. code:: bash

   #!/bin/bash
   set -o errexit

   _SRC=$1
   _DEST=$2
   _WORKERS=$3

   # Clone the dataset structure (not the data itself) locally so HF can find the
   # cache hashes it looks for. Else HF might think he needs to redo some
   # preprocessing. Directories will be created and symlinks will replace the files
   bash sh_utils.sh ln_files "${_SRC}" "${_DEST}" $_WORKERS

   # Copy the preprocessed dataset to compute node's local dataset cache dir so it
   # is close to the GPUs for faster training. Since HF can very easily change the
   # hash to reference a preprocessed dataset, we only copy the data for the
   # current preprocess pipeline.
   python3 get_dataset_cache_files.py | bash sh_utils.sh cp_files "${_SRC}" "${_DEST}" $_WORKERS


**get_dataset_cache_files.py**

.. code:: python

   """List to stdout the files of the dataset"""

   from pathlib import Path
   import sys

   import datasets

   from py_utils import (
       get_dataset_builder, get_num_workers, get_raw_datasets, get_tokenizer,
       preprocess_datasets
   )


   if __name__ == "__main__":
       # Redirect outputs to stderr to avoid noize in stdout
       _stdout = sys.stdout
       sys.stdout = sys.stderr

       try:
           _CACHE_DIR = sys.argv[1]
       except IndexError:
           _CACHE_DIR = datasets.config.HF_DATASETS_CACHE
       try:
           _WORKERS = int(sys.argv[2])
       except IndexError:
           _WORKERS = get_num_workers()

       cache_dir = Path(_CACHE_DIR)
       builder = get_dataset_builder(cache_dir=_CACHE_DIR)
       raw_datasets = get_raw_datasets(builder)
       tokenizer = get_tokenizer()
       for dataset in preprocess_datasets(tokenizer, raw_datasets, num_workers=_WORKERS).values():
           for cache_file in dataset.cache_files:
               cache_file = Path(cache_file["filename"]).relative_to(cache_dir)
               print(cache_file, file=_stdout)


**Running this example**

.. code-block:: bash

   $ sbatch job.sh
