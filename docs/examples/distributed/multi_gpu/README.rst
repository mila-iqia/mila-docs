.. NOTE: This file is auto-generated from examples/distributed/multi_gpu/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

Multi-GPU Job
=============


Prerequisites:

* `examples/frameworks/pytorch_setup <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_
* `examples/distributed/single_gpu <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu>`_

Other interesting resources:

* `<https://sebarnold.net/dist_blog/>`_
* `<https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide>`_

Click here to see `the code for this example
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/multi_gpu>`_

**job.sh**

.. code:: diff

    # distributed/single_gpu/job.sh -> distributed/multi_gpu/job.sh
    #!/bin/bash
   -#SBATCH --ntasks=1
   -#SBATCH --ntasks-per-node=1
   +#SBATCH --ntasks=2
   +#SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=4
    #SBATCH --gpus-per-task=l40s:1
    #SBATCH --mem-per-gpu=16G
    #SBATCH --time=00:15:00

    # Exit on error
    set -e

    # Echo time and hostname into log
    echo "Date:     $(date)"
    echo "Hostname: $(hostname)"

    # To make your code as much reproducible as possible with
    # `torch.use_deterministic_algorithms(True)`, uncomment the following block:
    ## === Reproducibility ===
    ## Be warned that this can make your code slower. See
    ## https://pytorch.org/docs/stable/notes/randomness.html#cublas-and-cudnn-deterministic-operations
    ## for more details.
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8
    ## === Reproducibility (END) ===

    # Stage dataset into $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR/data
    cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
    # General-purpose alternatives combining copy and unpack:
    #     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
    #     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/

   -# Execute Python script
   +# Get a unique port for this job based on the job ID
   +export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
   +export MASTER_ADDR="127.0.0.1"
   +
   +# Execute Python script in each task (one per GPU)
    # Use the `--offline` option of `uv run` on clusters without internet access on compute nodes.
    # Using the `--locked` option can help make your experiments easier to reproduce (it forces
    # your uv.lock file to be up to date with the dependencies declared in pyproject.toml).
   -srun uv run python main.py
   +# --gres-flags=allow-task-sharing is required to allow tasks on the same node to
   +# access GPUs allocated to other tasks on that node. Without this flag,
   +# --gpus-per-task=1 would isolate each task to only see its own GPU, which
   +# causes a a mysterious NCCL error in
   +# nn.parallel.DistributedDataParallel:
   +# ncclUnhandledCudaError: Call to CUDA function failed.
   +# when NCCL tries to communicate to local GPUs via shared memory but fails due
   +# to cgroups isolation. See https://slurm.schedmd.com/srun.html#OPT_gres-flags
   +# and https://support.schedmd.com/show_bug.cgi?id=17875 for details.
   +srun --gres-flags=allow-task-sharing uv run python main.py

**pyproject.toml**

.. code:: toml

   [project]
   name = "multi-gpu-example"
   version = "0.1.0"
   description = "Add your description here"
   readme = "README.rst"
   requires-python = ">=3.11,<3.14"
   dependencies = [
       "rich>=14.0.0",
       "torch>=2.7.1",
       "torchvision>=0.22.1",
       "tqdm>=4.67.1",
   ]

**main.py**

.. code:: diff

    # distributed/single_gpu/main.py -> distributed/multi_gpu/main.py
   -"""Single-GPU training example."""
   +"""Multi-GPU Training example."""

    import argparse
    import logging
    import os
    import random
    import sys
    from pathlib import Path

    import numpy as np
    import rich.logging
    import torch
   +import torch.distributed
    from torch import Tensor, nn
   +from torch.distributed import ReduceOp
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, random_split
   +from torch.utils.data.distributed import DistributedSampler
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from torchvision.models import resnet18
    from tqdm import tqdm


    # To make your code as much reproducible as possible, uncomment the following
    # block:
    ## === Reproducibility ===
    ## Be warned that this can make your code slower. See
    ## https://pytorch.org/docs/stable/notes/randomness.html#cublas-and-cudnn-deterministic-operations
    ## for more details.
    # torch.use_deterministic_algorithms(True)
    ## === Reproducibility (END) ===


    def main():
        # Use an argument parser so we can pass hyperparameters from the command line.
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--learning-rate", type=float, default=5e-4)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()

        epochs: int = args.epochs
        learning_rate: float = args.learning_rate
        weight_decay: float = args.weight_decay
   +    # NOTE: This is the "local" batch size, per-GPU.
        batch_size: int = args.batch_size
        seed: int = args.seed

        # Seed the random number generators as early as possible for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Check that the GPU is available
        assert torch.cuda.is_available() and torch.cuda.device_count() > 0
   +    rank, world_size = setup()
   +    is_master = rank == 0
   +    # When using --gpus-per-task=1, SLURM sets CUDA_VISIBLE_DEVICES so each process
   +    # only sees one GPU (index 0). Use device 0 directly.
        device = torch.device("cuda", 0)

        # Setup logging (optional, but much better than using print statements)
        # Uses the `rich` package to make logs pretty.
        logging.basicConfig(
            level=logging.INFO,
   -        format="%(message)s",
   +        format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
            handlers=[
                rich.logging.RichHandler(
                    markup=True,
                    console=rich.console.Console(
                        # Allower wider log lines in sbatch output files than on the terminal.
                        width=120 if not sys.stdout.isatty() else None
                    ),
                )
            ],
        )

        logger = logging.getLogger(__name__)
   +    logger.info(f"World size: {world_size}, global rank: {rank}")

        # Create a model and move it to the GPU.
        model = resnet18(num_classes=10)
        model.to(device=device)

   +    # Wrap the model with DistributedDataParallel
   +    # (See https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
   +    # When using --gpus-per-task=1, each process only sees one GPU (index 0).
   +    model = nn.parallel.DistributedDataParallel(
   +        model, device_ids=[0], output_device=0
   +    )
   +
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Setup CIFAR10
        num_workers = get_num_workers()
        dataset_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "data"
   -    train_dataset, valid_dataset, test_dataset = make_datasets(str(dataset_path))
   +    train_dataset, valid_dataset, test_dataset = make_datasets(
   +        str(dataset_path), is_master=is_master
   +    )
   +
   +    # Restricts data loading to a subset of the dataset exclusive to the current process
   +    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
   +    valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)
   +    test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
   +
   +    # NOTE: Here `batch_size` is still the "local" (per-gpu) batch size.
   +    # This way, the effective batch size scales directly with number of GPUs, no need to specify it
   +    # in advance. You might want to adjust the learning rate and other hyper-parameters though.
   +    if is_master:
   +        logger.info(f"Effective batch size: {batch_size * world_size}")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
   -        shuffle=True,
   +        shuffle=False,  # shuffling is now done in the sampler, not the dataloader.
   +        sampler=train_sampler,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
   +        sampler=valid_sampler,
        )
        _test_dataloader = DataLoader(  # NOTE: Not used in this example.
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
   +        sampler=test_sampler,
        )

        # Checkout the "checkpointing and preemption" example for more info!
        logger.debug("Starting training from scratch.")

        for epoch in range(epochs):
            logger.debug(f"Starting epoch {epoch}/{epochs}")

   +        # NOTE: Here we need to call `set_epoch` so the ordering is able to change at each epoch.
   +        train_sampler.set_epoch(epoch)
   +
            # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
            model.train()

            # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Train epoch {epoch}",
   -            disable=not sys.stdout.isatty(),  # Disable progress bar in non-interactive environments.
   +            # Disable progress bar in non-interactive environments.
   +            disable=not (sys.stdout.isatty() and is_master),
            )

            # Training loop
            for batch in train_dataloader:
                # Move the batch to the GPU before we pass it to the model
                batch = tuple(item.to(device) for item in batch)
                x, y = batch

                # Forward pass
                logits: Tensor = model(x)

   -            loss = F.cross_entropy(logits, y)
   +            local_loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
   -            loss.backward()
   +            local_loss.backward()
   +            # NOTE: nn.DistributedDataParallel automatically averages the gradients across devices.
                optimizer.step()

                # Calculate some metrics:
   -            n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
   -            n_samples = y.shape[0]
   +            # local metrics
   +            local_n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
   +            local_n_samples = logits.shape[0]
   +            local_accuracy = local_n_correct_predictions / local_n_samples
   +
   +            # "global" metrics: calculated with the results from all workers
   +            # NOTE: Creating new tensors to hold the "global" values, but this isn't required.
   +            n_correct_predictions = local_n_correct_predictions.clone()
   +            # Reduce the local metrics across all workers, sending the result to rank 0.
   +            torch.distributed.reduce(n_correct_predictions, dst=0, op=ReduceOp.SUM)
   +            # Actual (global) batch size for this step.
   +            n_samples = torch.as_tensor(local_n_samples, device=device)
   +            torch.distributed.reduce(n_samples, dst=0, op=ReduceOp.SUM)
   +            # Will store the average loss across all workers.
   +            loss = local_loss.clone()
   +            torch.distributed.reduce(loss, dst=0, op=ReduceOp.SUM)
   +            loss.div_(world_size)  # Report the average loss across all workers.
   +
                accuracy = n_correct_predictions / n_samples

   -            logger.debug(f"Accuracy: {accuracy.item():.2%}")
   -            logger.debug(f"Average Loss: {loss.item()}")
   +            logger.debug(f"(local) Accuracy: {local_accuracy:.2%}")
   +            logger.debug(f"(local) Loss: {local_loss.item()}")
   +            # NOTE: This would log the same values in all workers. Only logging on master:
   +            if is_master:
   +                logger.debug(f"Accuracy: {accuracy.item():.2%}")
   +                logger.debug(f"Average Loss: {loss.item()}")

                # Advance the progress bar one step and update the progress bar text.
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
            progress_bar.close()

            val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
   -        logger.info(
   -            f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}"
   -        )
   +        # NOTE: This would log the same values in all workers. Only logging on master:
   +        if is_master:
   +            logger.info(
   +                f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}"
   +            )

        print("Done!")


    @torch.no_grad()
    def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
        model.eval()

   -    total_loss = 0.0
   -    n_samples = 0
   -    correct_predictions = 0
   +    total_loss = torch.as_tensor(0.0, device=device)
   +    n_samples = torch.as_tensor(0, device=device)
   +    correct_predictions = torch.as_tensor(0, device=device)

        for batch in dataloader:
            batch = tuple(item.to(device) for item in batch)
            x, y = batch

            logits: Tensor = model(x)
            loss = F.cross_entropy(logits, y)

            batch_n_samples = x.shape[0]
            batch_correct_predictions = logits.argmax(-1).eq(y).sum()

   -        total_loss += loss.item()
   +        total_loss += loss
            n_samples += batch_n_samples
            correct_predictions += batch_correct_predictions

   +    # Sum up the metrics we gathered on each worker before returning the overall val metrics.
   +    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
   +    torch.distributed.all_reduce(correct_predictions, op=torch.distributed.ReduceOp.SUM)
   +    torch.distributed.all_reduce(n_samples, op=torch.distributed.ReduceOp.SUM)
   +
        accuracy = correct_predictions / n_samples
        return total_loss, accuracy


   +def setup():
   +    assert torch.distributed.is_available()
   +    print("PyTorch Distributed available.")
   +    print("  Backends:")
   +    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
   +    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
   +    print(f"    MPI:  {torch.distributed.is_mpi_available()}")
   +
   +    # DDP Job is being run via `srun` on a slurm cluster.
   +    rank = int(os.environ["SLURM_PROCID"])
   +    world_size = int(os.environ["SLURM_NTASKS"])
   +
   +    # SLURM var -> torch.distributed vars in case needed
   +    # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
   +    # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
   +    os.environ["RANK"] = str(rank)
   +    os.environ["WORLD_SIZE"] = str(world_size)
   +
   +    torch.distributed.init_process_group(
   +        backend="nccl",
   +        init_method="env://",
   +        world_size=world_size,
   +        rank=rank,
   +    )
   +    return rank, world_size
   +
   +
    def make_datasets(
        dataset_path: str,
   +    is_master: bool,
        val_split: float = 0.1,
        val_split_seed: int = 42,
    ):
        """Returns the training, validation, and test splits for CIFAR10.

        NOTE: We don't use image transforms here for simplicity.
        Having different transformations for train and validation would complicate things a bit.
        Later examples will show how to do the train/val/test split properly when using transforms.
   +
   +    NOTE: Only the master process (rank-0) downloads the dataset if necessary.
        """
   +    # - Master: Download (if necessary) THEN Barrier
   +    # - others: Barrier THEN *NO* Download
   +    if not is_master:
   +        # Wait for the master process to finish downloading (reach the barrier below)
   +        torch.distributed.barrier()
        train_dataset = CIFAR10(
   -        root=dataset_path, transform=transforms.ToTensor(), download=True, train=True
   +        root=dataset_path,
   +        transform=transforms.ToTensor(),
   +        download=is_master,
   +        train=True,
        )
        test_dataset = CIFAR10(
   -        root=dataset_path, transform=transforms.ToTensor(), download=True, train=False
   +        root=dataset_path,
   +        transform=transforms.ToTensor(),
   +        download=is_master,
   +        train=False,
        )
   +    if is_master:
   +        # Join the workers waiting in the barrier above. They can now load the datasets from disk.
   +        torch.distributed.barrier()
        # Split the training dataset into a training and validation set.
        n_samples = len(train_dataset)
        n_valid = int(val_split * n_samples)
        n_train = n_samples - n_valid
        train_dataset, valid_dataset = random_split(
            train_dataset, (n_train, n_valid), torch.Generator().manual_seed(val_split_seed)
        )
        return train_dataset, valid_dataset, test_dataset


    def get_num_workers() -> int:
        """Gets the optimal number of DatLoader workers to use in the current job."""
        if "SLURM_CPUS_PER_TASK" in os.environ:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
        return torch.multiprocessing.cpu_count()


    if __name__ == "__main__":
        main()


**Running this example**

.. code-block:: bash

    $ sbatch job.sh
