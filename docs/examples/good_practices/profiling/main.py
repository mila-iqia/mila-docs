import argparse
import json
import logging
import os
import time
from pathlib import Path

import rich.logging
import torch
import wandb
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm import tqdm


def main():
    # Use an argument parser so we can pass hyperparameters from the command line.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of data loader workers"
    )
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--use-wandb", action="store_true", help="Log with Weights and Biases"
    )
    parser.add_argument(
        "--wandb-user", type=str, default=None, help="Weights and Biases user"
    )
    parser.add_argument("--wandb-project", type=str, default="imagenet_profiling")
    parser.add_argument("--wandb-api-key", type=str, default="")
    args = parser.parse_args()

    skip_training: bool = args.skip_training
    num_workers: int = args.num_workers
    n_samples: int = args.n_samples
    batch_size: int = args.batch_size
    epochs: int = args.epochs
    learning_rate: float = args.learning_rate
    weight_decay: float = args.weight_decay
    use_wandb: bool = args.use_wandb
    wandb_user: str = args.wandb_user
    wandb_project: str = args.wandb_project
    wandb_api_key: str = args.wandb_api_key

    if use_wandb:
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, entity=wandb_user, config=vars(args))

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)

    # Setup logging (optional, but much better than using print statements)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            rich.logging.RichHandler(markup=True)
        ],  # Very pretty, uses the `rich` package.
    )

    logger = logging.getLogger(__name__)

    # Create a model and move it to the GPU.
    model = resnet50(num_classes=1000)
    model.to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Setup ImageNet
    logger.info("Setting up ImageNet")
    num_workers = get_num_workers()
    dataset_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "imagenet"
    train_dataset, valid_dataset, test_dataset = make_datasets(
        str(dataset_path), n_samples=n_samples
    )
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

    for epoch in range(epochs):
        logger.debug(f"Starting epoch {epoch}/{epochs}")
        # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
        model.train()
        # NOTE: using a progress bar from tqdm because it's nicer than using `print`.

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Train epoch {epoch}",
            # hint: look at unit_scale and unit params
            unit="Samples",
            unit_scale=True,
        )

        # Training loop
        start_time = time.time()
        num_samples = 0
        num_updates = 0
        for batch in progress_bar:
            # Move the batch to the GPU before we pass it to the model
            batch = tuple(item.to(device) for item in batch)
            x, y = batch
            num_samples += x.shape[0]
            if skip_training:
                continue
            # Forward pass
            logits: Tensor = model(x)

            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_updates += 1

            # Calculate some metrics:
            n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
            n_samples = y.shape[0]
            accuracy = n_correct_predictions / n_samples

            logger.debug(f"Accuracy: {accuracy.item():.2%}")
            logger.debug(f"Average Loss: {loss.item()}")

            # Log metrics with wandb
            if use_wandb:
                wandb.log({"accuracy": accuracy.item(), "loss": loss.item()})

            # Advance the progress bar one step and update the progress bar text.
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())

        elapsed_time = time.time() - start_time
        samples_per_second = num_samples / elapsed_time
        updates_per_second = num_updates / elapsed_time

        val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)

        logger.info(
            f"epoch {epoch}: samples/s: {samples_per_second},"
            f"updates/s: {updates_per_second}, "
            f"val_loss: {val_loss:.3f}, val_accuracy: {val_accuracy:.2%}"
        )
        if use_wandb:
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

    print(
        json.dumps(
            {
                "samples/s": samples_per_second,
                "updates/s": updates_per_second,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
    )


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
    return total_loss, float(accuracy)


def make_datasets(
    dataset_path: str,
    n_samples: int | None = None,
    val_split: float = 0.1,
    val_split_seed: int = 42,
    target_size: tuple = (224, 224),
):
    """Returns the training, validation, and test splits for ImageNet.

    NOTE: We don't use image transforms here for simplicity.
    Having different transformations for train and validation would complicate things a bit.
    Later examples will show how to do the train/val/test split properly when using transforms.
    """

    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "val")

    generator = torch.Generator().manual_seed(val_split_seed)
    # get the trans
    train_transform = Compose(
        [
            RandomResizedCrop(target_size),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transform = Compose(
        [
            Resize(target_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(
        root=train_dir,
        transform=train_transform,
    )
    # take a subset of n_samples of train_dataset (indices at random)

    if n_samples is not None and n_samples > 0:
        gen = torch.Generator().manual_seed(val_split_seed)

        train_dataset = Subset(  # todo: use the generator keyword to make this deterministic
            train_dataset,
            indices=torch.randperm(len(train_dataset), generator=gen)[
                :n_samples
            ].tolist(),
        )

    test_dataset = ImageFolder(
        root=test_dir,
        transform=val_test_transform,
    )

    # Split the training dataset into training and validation
    _n_samples = len(train_dataset)
    n_valid = int(val_split * _n_samples)
    n_train = _n_samples - n_valid

    train_dataset, valid_dataset = random_split(
        train_dataset,
        [n_train, n_valid],
        generator=generator,
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
