.. NOTE: This file is auto-generated from examples/good_practices/hpo_with_orion/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

.. _hpo_with_orion:

Hyperparameter Optimization with Oríon
======================================

There are frameworks that allow to do hyperparameter optimization, like
`wandb <https://wandb.ai/>`_,
and `Oríon <https://orion.readthedocs.io/en/stable/index.html>`_.
Here we provide an example for Oríon, the HPO framework developped at Mila.

**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* `examples/frameworks/pytorch_setup <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_

The full documentation for Oríon is available `on Oríon's ReadTheDocs page
<https://orion.readthedocs.io/en/stable/index.html>`_.


The full source code for this example is available on `the mila-docs GitHub repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/hpo_with_orion>`_


**job.sh**

.. code:: diff

    # distributed/single_gpu/job.sh -> good_practices/hpo_with_orion/job.sh
    #!/bin/bash
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    #SBATCH --time=00:15:00

    set -e  # exit on error.
    echo "Date:     $(date)"
    echo "Hostname: $(hostname)"

    # Stage dataset into $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR/data
    cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
    # General-purpose alternatives combining copy and unpack:
    #     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
    #     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/

   -# Execute Python script
   -# Use `uv run --offline` on clusters without internet access on compute nodes.
   -uv run python main.py
   +# =============
   +# Execute Orion
   +# =============
   +
   +# Specify an experiment name with `-n`,
   +# which could be reused to display results (see section "Running example" below)
   +
   +# Specify max trials (here 10) to prevent a too-long run.
   +
   +# Then you can specify a search space for each `main.py`'s script parameter
   +# you want to optimize. Here we optimize only the learning rate.
   +
   +orion hunt -n orion-example --exp-max-trials 10 uv run python main.py --learning-rate~'loguniform(1e-5, 1.0)'

**pyproject.toml**

.. code:: toml

   [project]
   name = "orion-example"
   version = "0.1.0"
   description = "Add your description here"
   readme = "README.md"
   requires-python = ">=3.11"
   dependencies = [
       "numpy>=2.3.1",
       "orion>=0.2.7",
       "rich>=14.0.0",
       "torch>=2.7.1",
       "torchvision>=0.22.1",
       "tqdm>=4.67.1",
   ]

**main.py**

.. code:: diff

    # distributed/single_gpu/main.py -> good_practices/hpo_with_orion/main.py
   -"""Single-GPU training example."""
   +"""Hyperparameter optimization using Oríon."""

    import argparse
   +import json
    import logging
    import os
    from pathlib import Path
    import sys

    import rich.logging
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from torchvision.models import resnet18
    from tqdm import tqdm

   +from orion.client import report_objective
   +

    def main():
   -    # Use an argument parser so we can pass hyperparameters from the command line.
   +    # Add an argument parser so that we can pass hyperparameters from command line.
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--learning-rate", type=float, default=5e-4)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--batch-size", type=int, default=128)
        args = parser.parse_args()

   -    epochs: int = args.epochs
   -    learning_rate: float = args.learning_rate
   -    weight_decay: float = args.weight_decay
   -    batch_size: int = args.batch_size
   +    epochs = args.epochs
   +    learning_rate = args.learning_rate
   +    weight_decay = args.weight_decay
   +    batch_size = args.batch_size

        # Check that the GPU is available
        assert torch.cuda.is_available() and torch.cuda.device_count() > 0
        device = torch.device("cuda", 0)

        # Setup logging (optional, but much better than using print statements)
        # Uses the `rich` package to make logs pretty.
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
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

   +    logger.info(f"Args: {json.dumps(vars(args), indent=1)}")
   +
        # Create a model and move it to the GPU.
        model = resnet18(num_classes=10)
        model.to(device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

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
                disable=not sys.stdout.isatty(),  # Disable progress bar in non-interactive environments.
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
            logger.info(
                f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}"
            )

   +    # We report to Orion the objective that we want to minimize.
   +    report_objective(1 - val_accuracy.item())
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

Oríon must be installed inside the "pytorch" environment using following command:

.. code-block:: bash

    pip install orion

Exit the interactive job once the environment has been created and Oríon installed.
You can then launch the example:

.. code-block:: bash

    $ sbatch job.sh

To get more information about the optimization run, activate "pytorch" environment
and run ``orion info`` with the experiment name:

.. code-block:: bash

    $ conda activate pytorch
    $ orion info -n orion-example

You can also generate a plot to visualize the optimization run. For example:

.. code-block:: bash

    $ orion plot regret -n orion-example

For more complex and useful plots, see `Oríon documentation
<https://orion.readthedocs.io/en/stable/auto_examples/plot_4_partial_dependencies.html>`_.
