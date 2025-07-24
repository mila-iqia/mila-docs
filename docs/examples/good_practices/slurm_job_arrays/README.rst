.. NOTE: This file is auto-generated from examples/good_practices/slurm_job_arrays/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

.. _slurm_job_arrays:

Launch many jobs using SLURM job arrays
=======================================

Sometimes you may want to run many tasks by changing just a single parameter.

One way to do that is to use SLURM job arrays, which consists of launching an array of jobs using the same script.
Each job will run with a specific environment variable called ``SLURM_ARRAY_TASK_ID``, containing the job index value inside job array.
You can then slightly modify your script to choose appropriate parameter based on this variable.

You can find more info about job arrays in the `SLURM official documentation page <https://slurm.schedmd.com/job_array.html>`_.


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* `examples/frameworks/pytorch_setup <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/slurm_job_arrays>`_


**main.py**

.. code:: diff

    # distributed/single_gpu/main.py -> good_practices/slurm_job_arrays/main.py
    """Single-GPU training example."""
   -
   -import argparse
    import logging
    import os
    from pathlib import Path
   -import sys
   +import argparse

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


    def main():
   +    # Use SLURM ARRAY TASK ID to seed a random number generator.
   +    # This way, each job in the job array will have different hyper-parameters.
   +    in_job_array = "SLURM_ARRAY_TASK_ID" in os.environ
   +    if in_job_array:
   +        array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
   +        array_task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
   +        print(f"This job is at index {array_task_id} in a job array of size {array_task_count}")
   +
   +        gen = numpy.random.default_rng(seed=array_task_id)
   +        # Use random number generator to generate the default values of hyper-parameters.
   +        # If a value is passed from the command-line, it will override this and be used instead.
   +        default_learning_rate = gen.uniform(1e-6, 1e-2)
   +        default_weight_decay = gen.uniform(1e-6, 1e-3)
   +        default_batch_size = gen.integers(16, 256)
   +    else:
   +        default_learning_rate = 5e-4
   +        default_weight_decay = 1e-4
   +        default_batch_size = 128
   +
        # Use an argument parser so we can pass hyperparameters from the command line.
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--epochs", type=int, default=10)
   -    parser.add_argument("--learning-rate", type=float, default=5e-4)
   -    parser.add_argument("--weight-decay", type=float, default=1e-4)
   -    parser.add_argument("--batch-size", type=int, default=128)
   +    parser.add_argument("--learning-rate", type=float, default=default_learning_rate)
   +    parser.add_argument("--weight-decay", type=float, default=default_weight_decay)
   +    parser.add_argument("--batch-size", type=int, default=default_batch_size)
        args = parser.parse_args()

        epochs: int = args.epochs
        learning_rate: float = args.learning_rate
        weight_decay: float = args.weight_decay
        batch_size: int = args.batch_size

        # Check that the GPU is available
        assert torch.cuda.is_available() and torch.cuda.device_count() > 0
        device = torch.device("cuda", 0)

        # Setup logging (optional, but much better than using print statements)
   -    # Uses the `rich` package to make logs pretty.
        logging.basicConfig(
            level=logging.INFO,
   -        format="%(message)s",
   -        handlers=[
   -            rich.logging.RichHandler(
   -                markup=True,
   -                console=rich.console.Console(
   -                    # Allower wider log lines in sbatch output files than on the terminal.
   -                    width=120 if not sys.stdout.isatty() else None
   -                ),
   -            )
   -        ],
   +        handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
        )

        logger = logging.getLogger(__name__)

        # Create a model and move it to the GPU.
        model = resnet18(num_classes=10)
        model.to(device=device)

   -    optimizer = torch.optim.AdamW(
   -        model.parameters(), lr=learning_rate, weight_decay=weight_decay
   -    )
   +    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Setup CIFAR10
        num_workers = get_num_workers()
        dataset_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "data"
        train_dataset, valid_dataset, test_dataset = make_datasets(str(dataset_path))
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

        for epoch in range(epochs):
            logger.debug(f"Starting epoch {epoch}/{epochs}")

            # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
            model.train()

            # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Train epoch {epoch}",
   -            disable=not sys.stdout.isatty(),  # Disable progress bar in non-interactive environments.
            )

            # Training loop
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

                # Advance the progress bar one step and update the progress bar text.
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
            progress_bar.close()

            val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
   -        logger.info(
   -            f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}"
   -        )
   +        logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

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
            batch_correct_predictions = logits.argmax(-1).eq(y).sum()

            total_loss += loss.item()
            n_samples += batch_n_samples
            correct_predictions += batch_correct_predictions

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

This assumes you already created a conda environment named "pytorch" as in
Pytorch example:

* :ref:`pytorch_setup`

Exit the interactive job once the environment has been created.
You can then launch a job array using ``sbatch`` argument ``--array``.

.. code-block:: bash

    $ sbatch --array=1-5 job.sh


In this example, 5 jobs will be launched with indices (therefore, values of ``SLURM_ARRAY_TASK_ID``) from 1 to 5.
