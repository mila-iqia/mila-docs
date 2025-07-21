"""Hyperparameter optimization using Oríon."""

import argparse
import json
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

from orion.client import report_objective


def main():
    # Add an argument parser so that we can pass hyperparameters from command line.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size

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

    logger.info(f"Args: {json.dumps(vars(args), indent=1)}")

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

    # We report to Orion the objective that we want to minimize.
    report_objective(1 - val_accuracy.item())

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
