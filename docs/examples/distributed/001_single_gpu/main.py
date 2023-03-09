import os

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from tqdm import tqdm


def main():
    # Check GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)

    batch_size = 128
    training_epochs = 5
    num_workers = get_num_workers()

    # Obtain CIFAR10
    dataset_path = os.environ.get("SLURM_TMPDIR", "../dataset")
    training_dataset, validation_dataset, test_dataset = make_datasets(dataset_path)

    # Create a model and move it to the GPU.
    model = resnet18(num_classes=10)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    training_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, num_workers=num_workers, batch_size=batch_size
    )
    # NOTE: Not used in this example
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size)

    batches_per_epoch = len(training_dataloader)
    for epoch in range(training_epochs):
        print(f"Starting epoch {epoch}/{training_epochs}")

        # Set the model in training mode (this is important for e.g. BatchNorm and Dropout layers)
        model.train()

        # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
        progress_bar = tqdm(total=batches_per_epoch, desc=f"Training epoch {epoch}")

        # Training loop
        for batch in training_dataloader:
            # Move the batch to the GPU before we pass it to the model
            batch = tuple(item.to(device) for item in batch)
            x, y = batch

            # Forward pass
            logits = model(x)

            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
            progress_bar.update(1)

            accuracy = logits.detach().argmax(-1).eq(y).float().mean()
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
        progress_bar.close()

        val_loss, val_accuracy = validation_loop(model, validation_dataloader, device)
        print(f"Epoch {epoch}: Validation loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

    print("Done!")


@torch.no_grad()
def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for batch in dataloader:
        batch = tuple(item.to(device) for item in batch)
        x, y = batch
        logits: Tensor = model(x)
        loss = F.cross_entropy(logits, y)

        val_loss += loss.item()
        correct_predictions += logits.argmax(-1).eq(y).sum().item()
        total_predictions += y.shape[0]

    val_accuracy = correct_predictions / total_predictions
    return val_loss, val_accuracy


def make_datasets(dataset_path: str, val_split: float = 0.1, val_split_seed: int = 42):
    # We don't use image transforms here for simplicity.
    # Having different transformations for train and validation would complicate things a bit.
    # Later examples will show how to do the train/val/test split properly when using transforms.
    train_dataset = CIFAR10(root=dataset_path, transform=ToTensor(), train=True)
    test_dataset = CIFAR10(root=dataset_path, transform=ToTensor(), train=False)
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


if __name__ == "__main__":
    main()
