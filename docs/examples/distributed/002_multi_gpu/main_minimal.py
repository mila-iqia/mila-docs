import os

import torch
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

    # Obtain CIFAR10
    dataset_path = os.environ.get("SLURM_TMPDIR", "../dataset")

    # NOTE: We don't use image transforms here for simplicity.
    dataset = CIFAR10(root=dataset_path, transform=ToTensor(), train=True)
    val_split = 0.1
    training_dataset, validation_dataset = random_split(dataset, ((1 - val_split), val_split))

    # Create a model and move it to the GPU.
    model = resnet18(num_classes=10)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 128
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    training_epochs = 1
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

        # Validation loop
        model.eval()
        epoch_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for batch in validation_dataloader:
            batch = tuple(item.to(device) for item in batch)
            x, y = batch
            with torch.no_grad():
                logits = model(x)
                validation_loss = F.cross_entropy(logits, y)

            epoch_val_loss += validation_loss.item()
            correct_predictions += logits.argmax(-1).eq(y).sum().item()
            total_predictions += y.shape[0]

        validation_accuracy = correct_predictions / total_predictions
        print(
            f"Epoch {epoch}: Validation loss: {epoch_val_loss:.3f} accuracy: {validation_accuracy:.2%}"
        )

    print("Done!")
    # NOTE: You could save


if __name__ == "__main__":
    main()
