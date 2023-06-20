"""Single-GPU training example."""
import logging
import os
from pathlib import Path
import random
import shutil
import numpy

import rich.logging
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm
from logging import getLogger as get_logger


SCRATCH = Path(os.environ["SCRATCH"])
SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
SLURM_JOBID = int(os.environ["SLURM_JOBID"])

_CHECKPOINTS_DIR = SCRATCH / "checkpoints"

logger = get_logger(__name__)



def main():
    training_epochs = 10
    learning_rate = 5e-4
    weight_decay = 1e-4
    batch_size = 128
    checkpoint_file = _CHECKPOINTS_DIR / SLURM_JOBID / "checkpoint.pth"
    start_epoch = 0
    best_acc = 0

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)

    # Setup logging (optional, but much better than using print statements)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
    )

    # Create a model.
    model = resnet18(num_classes=10)

    # Move the model to the GPU.
    model.to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Resume from a checkpoint
    if "SLURM_RESTART_COUNT" in os.environ:
        restart_count = int(os.environ["SLURM_RESTART_COUNT"])
        logger.info(f"This job has been restarted {restart_count} times.")

    if checkpoint_file.exists():
        logger.debug(f"loading checkpoint '{checkpoint_file}'")
        # Map model to be loaded to gpu.
        checkpoint = torch.load(checkpoint_file, map_location=device)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        # best_acc may be from a checkpoint from a different GPU
        best_acc = best_acc.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        random.setstate(checkpoint["random_state"])
        numpy.random.set_state(checkpoint["numpy_random_state"])
        torch.random.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.random.set_rng_state_all(checkpoint["torch_cuda_random_state"])

        logger.info(f"loaded checkpoint '{checkpoint_file}' (Starting at epoch {start_epoch})")
    else:
        logger.info(f"no checkpoint found at '{checkpoint_file}', starting training from scratch.")

    # Setup CIFAR10
    num_workers = get_num_workers()
    dataset_path = (SLURM_TMPDIR or Path("..")) / "data"

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

    logger.debug("Starting training from scratch.")

    for epoch in range(start_epoch, training_epochs):
        logger.debug(f"Starting epoch {epoch}/{training_epochs}")

        # Set the model in training mode (this is important for e.g. BatchNorm and Dropout layers)
        model.train()

        # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Train epoch {epoch}",
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

            # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
        progress_bar.close()

        val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
        logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

        # remember best acc and save checkpoint
        is_best = val_accuracy > best_acc
        best_acc = max(val_accuracy, best_acc)

        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "random_state": random.getstate(),
                "numpy_random_state": numpy.random.get_state(),
                "torch_random_state": torch.random.get_rng_state(),
                "torch_cuda_random_state": torch.cuda.random.get_rng_state_all(),
            },
            checkpoint_file,
        )
        if is_best:
            shutil.copyfile(checkpoint_file, checkpoint_file.parent / "model_best.pth")

    print("Done!")


@torch.no_grad()
def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()

    total_loss = 0.0
    n_samples = 0
    correct_predictions = 0

    for batch in dataloader:
        batch: tuple[Tensor, Tensor] = tuple(item.to(device) for item in batch)
        x, y = batch

        logits: Tensor = model(x)
        loss = F.cross_entropy(logits, y)

        batch_n_samples = x.size(0)
        batch_correct_predictions = logits.argmax(-1).eq(y).sum().item()

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
    train_dataset, valid_dataset = random_split(
        train_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
    )
    return train_dataset, valid_dataset, test_dataset


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


def save_checkpoint(
    state: dict, is_best: bool, filename: str = f"{_CHECKPOINTS_DIR}/checkpoint.pth.tar"
) -> None:
    ...


def load_checkpoint(
    checkpoint_file: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
):
    # Map model to be loaded to gpu.
    checkpoint = torch.load(checkpoint_file, map_location=device)
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    random.setstate(checkpoint["random_state"])
    numpy.random.set_state(checkpoint["numpy_random_state"])
    torch.random.set_rng_state(checkpoint["torch_random_state"])
    torch.cuda.random.set_rng_state_all(checkpoint["torch_cuda_random_state"])
    return start_epoch, best_acc


if __name__ == "__main__":
    main()
