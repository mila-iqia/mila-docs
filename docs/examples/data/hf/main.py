"""HuggingFace training example."""
import logging

import rich.logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from py_utils import (
    get_dataset_builder, get_num_workers, get_raw_datasets, get_tokenizer,
    preprocess_datasets
)


def main():
    training_epochs = 1
    batch_size = 256

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)

    # Setup logging (optional, but much better than using print statements)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
    )

    logger = logging.getLogger(__name__)

    # Setup ImageNet
    num_workers = get_num_workers()
    train_dataset, valid_dataset, test_dataset = make_datasets(num_workers)
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

        # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Train epoch {epoch}",
        )

        # Training loop
        for batch in train_dataloader:
            # Move the batch to the GPU before we pass it to the model
            batch = {k:item.to(device) for k, item in batch.items()}

            # [Training of the model goes here]

            # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
            progress_bar.update(1)
        progress_bar.close()

        val_loss, val_accuracy = validation_loop(None, valid_dataloader, device)
        logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

    print("Done!")


@torch.no_grad()
def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
    total_loss = 0.0
    n_samples = 0
    correct_predictions = 0

    for batch in dataloader:
        batch = {k:item.to(device) for k, item in batch.items()}

        batch_n_samples = batch["input_ids"].data.shape[0]

        n_samples += batch_n_samples

    accuracy = correct_predictions / n_samples
    return total_loss, accuracy


def make_datasets(num_workers:int=None):
    """Returns the training, validation, and test splits for the prepared dataset.
    """
    builder = get_dataset_builder()
    raw_datasets = get_raw_datasets(builder)
    tokenizer = get_tokenizer()
    preprocessed_datasets = preprocess_datasets(tokenizer, raw_datasets, num_workers=num_workers)
    return (
        preprocessed_datasets["train"], preprocessed_datasets["validation"],
        preprocessed_datasets["test"]
    )


if __name__ == "__main__":
    main()
