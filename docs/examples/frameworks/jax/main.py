"""Single-GPU training example.

This Jax example is heavily based on the following examples:

* https://juliusruseckas.github.io/ml/flax-cifar10.html
* https://github.com/fattorib/Flax-ResNets/blob/master/main_flax.py
"""
import logging
import math
import os
from pathlib import Path
from typing import Any, Sequence

import PIL.Image
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rich.logging
import torch

from flax.training import train_state, common_utils
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from model import ResNet


class TrainState(train_state.TrainState):
    batch_stats: Any


class ToArray(torch.nn.Module):
    """convert image to float and 0-1 range"""
    dtype = np.float32

    def __call__(self, x):
        assert isinstance(x, PIL.Image.Image)
        x = np.asarray(x, dtype=self.dtype)
        x /= 255.0
        return x


def numpy_collate(batch: Sequence):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def main():
    training_epochs = 10
    learning_rate = 5e-4
    weight_decay = 1e-4
    batch_size = 128

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    rng = jax.random.PRNGKey(0)

    # Setup logging (optional, but much better than using print statements)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
    )

    logger = logging.getLogger(__name__)

    # Create a model.
    model = ResNet(
        10,
        channel_list = [64, 128, 256, 512],
        num_blocks_list = [2, 2, 2, 2],
        strides = [1, 1, 2, 2, 2],
        head_p_drop = 0.3
    )

    @jax.jit
    def initialize(params_rng, image_size=32):
        init_rngs = {'params': params_rng}
        input_shape = (1, image_size, image_size, 3)
        variables = model.init(init_rngs, jnp.ones(input_shape, jnp.float32), train=False)
        return variables

    # Setup CIFAR10
    num_workers = get_num_workers()
    dataset_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "data"
    train_dataset, valid_dataset, test_dataset = make_datasets(str(dataset_path))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=numpy_collate,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=numpy_collate,
    )
    test_dataloader = DataLoader(  # NOTE: Not used in this example.
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=numpy_collate,
    )

    train_steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    num_train_steps = train_steps_per_epoch * training_epochs
    shedule_fn = optax.cosine_onecycle_schedule(transition_steps=num_train_steps, peak_value=learning_rate)
    optimizer = optax.adamw(learning_rate=shedule_fn, weight_decay=weight_decay)

    params_rng, dropout_rng = jax.random.split(rng)
    variables = initialize(params_rng)

    state = TrainState.create(
        apply_fn = model.apply,
        params = variables['params'],
        batch_stats = variables['batch_stats'],
        tx = optimizer
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
        for input, target in train_dataloader:
            batch = {
                'image': input,
                'label': target,
            }
            state, loss, accuracy = train_step(state, batch, dropout_rng)

            logger.debug(f"Accuracy: {accuracy:.2%}")
            logger.debug(f"Average Loss: {loss}")

            # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss, accuracy=accuracy)
        progress_bar.close()

        val_loss, val_accuracy = validation_loop(state, valid_dataloader)
        logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

    print("Done!")


def cross_entropy_loss(logits, labels, num_classes=10):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    loss = jnp.mean(loss)
    return loss


@jax.jit
def train_step(state, batch, dropout_rng):
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(variables, batch['image'], train=True,
                                                 rngs={'dropout': dropout_rng}, mutable='batch_stats')
        loss = cross_entropy_loss(logits, batch['label'])
        accuracy = jnp.sum(jnp.argmax(logits, -1) == batch['label'])
        return loss, (accuracy, new_model_state)

    (loss, (accuracy, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    return new_state, loss, accuracy


@jax.jit
def validation_step(state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
    loss = cross_entropy_loss(logits, batch['label'])
    batch_correct_predictions = jnp.sum(jnp.argmax(logits, -1) == batch['label'])
    return loss, batch_correct_predictions


@torch.no_grad()
def validation_loop(state, dataloader: DataLoader):
    losses = []
    correct_predictions = []
    for input, target in dataloader:
        batch = {
            'image': input,
            'label': target,
        }
        loss, batch_correct_predictions = validation_step(state, batch)
        losses.append(loss)
        correct_predictions.append(batch_correct_predictions)

    total_loss = np.sum(losses)
    accuracy = np.mean(correct_predictions)
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
        root=dataset_path, transform=ToArray(), download=True, train=True
    )
    test_dataset = CIFAR10(
        root=dataset_path, transform=ToArray(), download=True, train=False
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
