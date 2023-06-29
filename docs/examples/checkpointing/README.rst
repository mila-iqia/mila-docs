Checkpointing
=============


**Prerequisites**

Make sure to read the following sections of the documentation before using this
example:

* :ref:`pytorch_setup`
* :ref:`001 - Single GPU Job`

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/data/checkpointing>`_


**job.sh**

.. code:: diff

    # distributed/001_single_gpu/job.sh -> checkpointing/job.sh
   old mode 100644
   new mode 100755
    #!/bin/bash
   -#SBATCH --gpus-per-task=rtx8000:1
   +#SBATCH --gpus-per-task=1
    #SBATCH --cpus-per-task=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --mem=16G
    #SBATCH --time=00:15:00
   +#SBATCH --requeue
   +#SBATCH --signal=B:TERM@300 # tells the controller to send SIGTERM to the job 5
   +                            # min before its time ends to give it a chance for
   +                            # better cleanup. If you cancel the job manually,
   +                            # make sure that you specify the signal as TERM like
   +                            # so `scancel --signal=TERM <jobid>`.
   +                            # https://dhruveshp.com/blog/2021/signal-propagation-on-slurm/
   +
   +# trap the signal to the main BATCH script here.
   +sig_handler()
   +{
   +    echo "BATCH interrupted"
   +    wait # wait for all children, this is important!
   +}
   +
   +trap 'sig_handler' SIGINT SIGTERM SIGCONT


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
    # conda install -y -n pytorch -c conda-forge rich tqdm

    # Activate pre-existing environment.
    conda activate pytorch


    # Stage dataset into $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR/data
   -cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
   +# Use --update to only copy newer files (since this might have already been executed)
   +cp --update /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
    # General-purpose alternatives combining copy and unpack:
    #     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
    #     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/


    # Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
    unset CUDA_VISIBLE_DEVICES

    # Execute Python script
   -python main.py
   +exec python main.py


**main.py**

.. code:: diff

    # distributed/001_single_gpu/main.py -> checkpointing/main.py
   -"""Single-GPU training example."""
   +"""Checkpointing example."""
   +from __future__ import annotations
   +
   +import contextlib
    import logging
    import os
   +import random
   +import shutil
   +from logging import getLogger as get_logger
    from pathlib import Path
   +from typing import Any, TypedDict

   +import numpy
    import rich.logging
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from torchvision.models import resnet18
    from tqdm import tqdm

   +SCRATCH = Path(os.environ["SCRATCH"])
   +SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
   +SLURM_JOBID = os.environ["SLURM_JOBID"]
   +
   +CHECKPOINT_FILE_NAME = "checkpoint.pth"
   +
   +logger = get_logger(__name__)
   +
   +
   +class RunState(TypedDict):
   +    """Typed dictionary containing the state of the training run which is saved at each epoch.
   +
   +    Using type hints helps prevent bugs and makes your code easier to read for both humans and
   +    machines (e.g. Copilot). This leads to less time spent debugging and better code suggestions.
   +    """
   +
   +    epoch: int
   +    best_acc: float
   +    model_state: dict[str, Tensor]
   +    optimizer_state: dict[str, Tensor]
   +
   +    random_state: tuple[Any, ...]
   +    numpy_random_state: dict[str, Any]
   +    torch_random_state: Tensor
   +    torch_cuda_random_state: list[Tensor]
   +

    def main():
   -    training_epochs = 10
   +    training_epochs = 5
        learning_rate = 5e-4
        weight_decay = 1e-4
        batch_size = 128
   +    run_dir = SCRATCH / "checkpointing_example" / SLURM_JOBID
   +    checkpoint_dir = run_dir / "checkpoints"
   +    random_seed: int = 123
   +    start_epoch: int = 0
   +    best_acc: float = 0.0

        # Check that the GPU is available
        assert torch.cuda.is_available() and torch.cuda.device_count() > 0
        device = torch.device("cuda", 0)

   +    # Seed the random number generators as early as possible.
   +    random.seed(random_seed)
   +    numpy.random.seed(random_seed)
   +    torch.random.manual_seed(random_seed)
   +    torch.cuda.manual_seed_all(random_seed)
   +
        # Setup logging (optional, but much better than using print statements)
        logging.basicConfig(
            level=logging.INFO,
   +        format="%(message)s",
            handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
        )

   -    logger = logging.getLogger(__name__)
   -
   -    # Create a model and move it to the GPU.
   +    # Create a model.
        model = resnet18(num_classes=10)
   +
   +    # Move the model to the GPU.
        model.to(device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

   -    # Setup CIFAR10
   +    # Try to resume from a checkpoint, if one exists.
   +    checkpoint: RunState | None = load_checkpoint(checkpoint_dir, map_location=device)
   +    if checkpoint:
   +        start_epoch = checkpoint["epoch"] + 1  # +1 to start at the next epoch.
   +        best_acc = checkpoint["best_acc"]
   +        model.load_state_dict(checkpoint["model_state"])
   +        optimizer.load_state_dict(checkpoint["optimizer_state"])
   +        random.setstate(checkpoint["random_state"])
   +        numpy.random.set_state(checkpoint["numpy_random_state"])
   +        # NOTE: Need to move those tensors to CPU before they can be loaded.
   +        torch.random.set_rng_state(checkpoint["torch_random_state"].cpu())
   +        torch.cuda.random.set_rng_state_all(t.cpu() for t in checkpoint["torch_cuda_random_state"])
   +        logger.info(f"Resuming training at epoch {start_epoch} (best_acc={best_acc:.2%}).")
   +    else:
   +        logger.info(f"No checkpoints found in {checkpoint_dir}. Training from scratch.")
   +
   +    # Setup the dataset
        num_workers = get_num_workers()
   -    dataset_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "data"
   +    dataset_path = (SLURM_TMPDIR or Path("..")) / "data"
   +
        train_dataset, valid_dataset, test_dataset = make_datasets(str(dataset_path))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
   +        # generator=torch.Generator().manual_seed(random_seed),
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
   +        # generator=torch.Generator().manual_seed(random_seed),
        )
        test_dataloader = DataLoader(  # NOTE: Not used in this example.
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

   -    # Checkout the "checkpointing and preemption" example for more info!
   -    logger.debug("Starting training from scratch.")
   -
   -    for epoch in range(training_epochs):
   +    for epoch in range(start_epoch, training_epochs):
            logger.debug(f"Starting epoch {epoch}/{training_epochs}")

   -        # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
   +        # Set the model in training mode (this is important for e.g. BatchNorm and Dropout layers)
            model.train()

   -        # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
   +        # NOTE: using a progress bar from tqdm much nicer than using `print`s).
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Train epoch {epoch}",
   +            unit_scale=train_dataloader.batch_size or 1,
   +            unit="samples",
            )

            # Training loop
   +        batch: tuple[Tensor, Tensor]
            for batch in train_dataloader:
                # Move the batch to the GPU before we pass it to the model
                batch = tuple(item.to(device) for item in batch)
                x, y = batch

                # Forward pass
                logits: Tensor = model(x)

                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate some metrics:
                n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
                n_samples = y.shape[0]
                accuracy = n_correct_predictions / n_samples

                logger.debug(f"Accuracy: {accuracy.item():.2%}")
                logger.debug(f"Average Loss: {loss.item()}")

   -            # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
   +            # Advance the progress bar one step, and update the text displayed in the progress bar.
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
            progress_bar.close()

            val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
            logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

   +        # remember best accuracy and save the current state.
   +        is_best = val_accuracy > best_acc
   +        best_acc = max(val_accuracy, best_acc)
   +
   +        if checkpoint_dir is not None:
   +            save_checkpoint(
   +                checkpoint_dir,
   +                is_best,
   +                RunState(
   +                    epoch=epoch,
   +                    model_state=model.state_dict(),
   +                    optimizer_state=optimizer.state_dict(),
   +                    random_state=random.getstate(),
   +                    numpy_random_state=numpy.random.get_state(legacy=False),
   +                    torch_random_state=torch.random.get_rng_state(),
   +                    torch_cuda_random_state=torch.cuda.random.get_rng_state_all(),
   +                    best_acc=best_acc,
   +                ),
   +            )
   +
        print("Done!")


    @torch.no_grad()
    def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
        model.eval()

        total_loss = 0.0
        n_samples = 0
        correct_predictions = 0

        for batch in dataloader:
            batch = tuple(item.to(device) for item in batch)
            x, y = batch

            logits: Tensor = model(x)
            loss = F.cross_entropy(logits, y)

            batch_n_samples = x.shape[0]
   -        batch_correct_predictions = logits.argmax(-1).eq(y).sum()
   +        batch_correct_predictions = logits.argmax(-1).eq(y).sum().item()

            total_loss += loss.item()
            n_samples += batch_n_samples
   -        correct_predictions += batch_correct_predictions
   +        correct_predictions += int(batch_correct_predictions)

        accuracy = correct_predictions / n_samples
        return total_loss, accuracy


    def make_datasets(
        dataset_path: str,
        val_split: float = 0.1,
        val_split_seed: int = 42,
    ):
        """Returns the training, validation, and test splits for CIFAR10.

        NOTE: We don't use image transforms here for simplicity.
        Having different transformations for train and validation would complicate things a bit.
        Later examples will show how to do the train/val/test split properly when using transforms.
        """
        train_dataset = CIFAR10(
            root=dataset_path, transform=transforms.ToTensor(), download=True, train=True
        )
        test_dataset = CIFAR10(
            root=dataset_path, transform=transforms.ToTensor(), download=True, train=False
        )
        # Split the training dataset into a training and validation set.
   -    n_samples = len(train_dataset)
   -    n_valid = int(val_split * n_samples)
   -    n_train = n_samples - n_valid
        train_dataset, valid_dataset = random_split(
   -        train_dataset, (n_train, n_valid), torch.Generator().manual_seed(val_split_seed)
   +        train_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
        )
        return train_dataset, valid_dataset, test_dataset


    def get_num_workers() -> int:
   -    """Gets the optimal number of DatLoader workers to use in the current job."""
   +    """Gets the optimal number of DataLoader workers to use in the current job."""
        if "SLURM_CPUS_PER_TASK" in os.environ:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
        return torch.multiprocessing.cpu_count()


   +def load_checkpoint(checkpoint_dir: Path, **torch_load_kwargs) -> RunState | None:
   +    """Loads the latest checkpoint if possible, otherwise returns `None`."""
   +    checkpoint_file = checkpoint_dir / CHECKPOINT_FILE_NAME
   +    backup = checkpoint_file.with_suffix(".backup")
   +
   +    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
   +    if restart_count:
   +        logger.info(f"NOTE: This job has been restarted {restart_count} times by SLURM.")
   +
   +    state: RunState | None = None
   +    if backup.exists():
   +        logger.debug(f"Job was interrupted while saving. Loading from the backup at {backup}")
   +        state = torch.load(checkpoint_file, **torch_load_kwargs)
   +    elif checkpoint_file.exists():
   +        # There is no backup file and the checkpoint file exists, so it should be good to load.
   +        logger.debug(f"Resuming from the checkpoint file at {checkpoint_file}")
   +        state = torch.load(checkpoint_file, **torch_load_kwargs)
   +    else:
   +        logger.debug(f"No checkpoint found in checkpoints dir ({checkpoint_dir}).")
   +        if restart_count:
   +            logger.warning(
   +                f"This job has been restarted {restart_count} times by SLURM, but no checkpoint "
   +                f"was found! This either means that your checkpointing code is broken, or that "
   +                "the job did not reach the checkpointing portion of your training loop."
   +            )
   +
   +    return state
   +
   +
   +def save_checkpoint(checkpoint_dir: Path, is_best: bool, state: RunState):
   +    """Saves a checkpoint with the current state of the run in the checkpoint dir.
   +
   +    The best checkpoint is also updated if `is_best` is `True`.
   +
   +    Parameters
   +    ----------
   +    checkpoint_dir: The checkpoint directory.
   +    is_best: Whether this is the best checkpoint so far.
   +    state: The dictionary containing all the things to save.
   +    """
   +    checkpoint_dir.mkdir(parents=True, exist_ok=True)
   +    checkpoint_file = checkpoint_dir / CHECKPOINT_FILE_NAME
   +
   +    # Make temporary backups of existing checkpoint files, in case our job gets interrupted while
   +    # saving, we can restart using the backups.
   +    with make_temporary_backup_if_exists(checkpoint_file):
   +        torch.save(state, checkpoint_file)
   +
   +    if is_best:
   +        best_checkpoint = checkpoint_file.with_name("model_best.pth")
   +        with make_temporary_backup_if_exists(best_checkpoint):
   +            shutil.copyfile(checkpoint_file, best_checkpoint)
   +
   +
   +@contextlib.contextmanager
   +def make_temporary_backup_if_exists(file: Path, backup: Path | None = None):
   +    """If the file exists, makes a temporary backup of it at `backup` and enters the "with" block.
   +
   +    Removes the backup when exiting the "with" block.
   +    """
   +    backup = backup or file.with_suffix(".backup")
   +    if file.exists():
   +        file.rename(backup)
   +    yield
   +    backup.unlink(missing_ok=True)
   +
   +
    if __name__ == "__main__":
        main()


**Running this example**

.. code-block:: bash

   $ sbatch job.sh
