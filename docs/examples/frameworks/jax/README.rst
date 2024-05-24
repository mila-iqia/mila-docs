.. NOTE: This file is auto-generated from examples/frameworks/jax/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

.. _jax:

Jax
===


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* `examples/frameworks/jax_setup <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/jax_setup>`_
* `examples/distributed/single_gpu <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/single_gpu>`_

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/jax>`_


**job.sh**

.. code:: diff

    # distributed/single_gpu/job.sh -> frameworks/jax/job.sh
   old mode 100755
   new mode 100644
    #!/bin/bash
    #SBATCH --gpus-per-task=rtx8000:1
    #SBATCH --cpus-per-task=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --mem=16G
    #SBATCH --time=00:15:00


    # Echo time and hostname into log
    echo "Date:     $(date)"
    echo "Hostname: $(hostname)"


    # Ensure only anaconda/3 module loaded.
    module --quiet purge
    # This example uses Conda to manage package dependencies.
    # See https://docs.mila.quebec/Userguide.html#conda for more information.
    module load anaconda/3
   -module load cuda/11.7

   -# Activate pre-existing environment.
   -# NOTE: Use the `make_env.sh` script to create the environment if you haven't already.
   -conda activate pytorch
   +# Creating the environment for the first time:
   +# conda create -y -n jax_ex -c "nvidia/label/cuda-11.8.0" cuda python=3.9 virtualenv pip
   +# conda activate jax_ex
   +# Install Jax using `pip`
   +# *Please note* that as soon as you install packages from `pip install`, you
   +# should not install any more packages using `conda install`
   +# pip install --upgrade "jax[cuda11_pip]" \
   +#    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   +# Other pip packages:
   +# pip install pillow optax rich torch torchvision flax tqdm
   +
   +# Activate the environment:
   +conda activate jax_ex


    # Stage dataset into $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR/data
    cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
    # General-purpose alternatives combining copy and unpack:
    #     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
    #     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/


   -# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
   -unset CUDA_VISIBLE_DEVICES
   -
    # Execute Python script
    python main.py


**main.py**

.. code:: diff

    # distributed/single_gpu/main.py -> frameworks/jax/main.py
   -"""Single-GPU training example."""
   +"""Single-GPU training example.
   +
   +This Jax example is heavily based on the following examples:
   +
   +* https://juliusruseckas.github.io/ml/flax-cifar10.html
   +* https://github.com/fattorib/Flax-ResNets/blob/master/main_flax.py
   +"""
    import argparse
    import logging
   +import math
    import os
    from pathlib import Path
   +from typing import Any, Sequence

   +import PIL.Image
   +import jax
   +import jax.numpy as jnp
   +import numpy as np
   +import optax
    import rich.logging
    import torch
   -from torch import Tensor, nn
   -from torch.nn import functional as F
   +
   +from flax.training import train_state, common_utils
    from torch.utils.data import DataLoader, random_split
   -from torchvision import transforms
    from torchvision.datasets import CIFAR10
   -from torchvision.models import resnet18
    from tqdm import tqdm

   +from model import ResNet
   +
   +
   +class TrainState(train_state.TrainState):
   +    batch_stats: Any
   +
   +
   +class ToArray(torch.nn.Module):
   +    """convert image to float and 0-1 range"""
   +    dtype = np.float32
   +
   +    def __call__(self, x):
   +        assert isinstance(x, PIL.Image.Image)
   +        x = np.asarray(x, dtype=self.dtype)
   +        x /= 255.0
   +        return x
   +
   +
   +def numpy_collate(batch: Sequence):
   +    if isinstance(batch[0], np.ndarray):
   +        return np.stack(batch)
   +    elif isinstance(batch[0], (tuple, list)):
   +        transposed = zip(*batch)
   +        return [numpy_collate(samples) for samples in transposed]
   +    else:
   +        return np.array(batch)
   +

    def main():
        # Use an argument parser so we can pass hyperparameters from the command line.
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--learning-rate", type=float, default=5e-4)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--batch-size", type=int, default=128)
        args = parser.parse_args()

        epochs: int = args.epochs
        learning_rate: float = args.learning_rate
        weight_decay: float = args.weight_decay
   +    # NOTE: This is the "local" batch size, per-GPU.
        batch_size: int = args.batch_size

        # Check that the GPU is available
        assert torch.cuda.is_available() and torch.cuda.device_count() > 0
   -    device = torch.device("cuda", 0)
   +    rng = jax.random.PRNGKey(0)

        # Setup logging (optional, but much better than using print statements)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
        )

        logger = logging.getLogger(__name__)

   -    # Create a model and move it to the GPU.
   -    model = resnet18(num_classes=10)
   -    model.to(device=device)
   +    # Create a model.
   +    model = ResNet(
   +        10,
   +        channel_list = [64, 128, 256, 512],
   +        num_blocks_list = [2, 2, 2, 2],
   +        strides = [1, 1, 2, 2, 2],
   +        head_p_drop = 0.3
   +    )

   -    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
   +    @jax.jit
   +    def initialize(params_rng, image_size=32):
   +        init_rngs = {'params': params_rng}
   +        input_shape = (1, image_size, image_size, 3)
   +        variables = model.init(init_rngs, jnp.ones(input_shape, jnp.float32), train=False)
   +        return variables

        # Setup CIFAR10
        num_workers = get_num_workers()
        dataset_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "data"
        train_dataset, valid_dataset, test_dataset = make_datasets(str(dataset_path))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
   +        collate_fn=numpy_collate,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
   +        collate_fn=numpy_collate,
        )
        test_dataloader = DataLoader(  # NOTE: Not used in this example.
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
   +        collate_fn=numpy_collate,
   +    )
   +
   +    train_steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
   +    num_train_steps = train_steps_per_epoch * epochs
   +    shedule_fn = optax.cosine_onecycle_schedule(transition_steps=num_train_steps, peak_value=learning_rate)
   +    optimizer = optax.adamw(learning_rate=shedule_fn, weight_decay=weight_decay)
   +
   +    params_rng, dropout_rng = jax.random.split(rng)
   +    variables = initialize(params_rng)
   +
   +    state = TrainState.create(
   +        apply_fn = model.apply,
   +        params = variables['params'],
   +        batch_stats = variables['batch_stats'],
   +        tx = optimizer
        )

        # Checkout the "checkpointing and preemption" example for more info!
        logger.debug("Starting training from scratch.")

        for epoch in range(epochs):
            logger.debug(f"Starting epoch {epoch}/{epochs}")

   -        # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
   -        model.train()
   -
            # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Train epoch {epoch}",
            )

            # Training loop
   -        for batch in train_dataloader:
   -            # Move the batch to the GPU before we pass it to the model
   -            batch = tuple(item.to(device) for item in batch)
   -            x, y = batch
   -
   -            # Forward pass
   -            logits: Tensor = model(x)
   -
   -            loss = F.cross_entropy(logits, y)
   +        for input, target in train_dataloader:
   +            batch = {
   +                'image': input,
   +                'label': target,
   +            }
   +            state, loss, accuracy = train_step(state, batch, dropout_rng)

   -            optimizer.zero_grad()
   -            loss.backward()
   -            optimizer.step()
   +            logger.debug(f"Accuracy: {accuracy:.2%}")
   +            logger.debug(f"Average Loss: {loss}")

   -            # Calculate some metrics:
   -            n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
   -            n_samples = y.shape[0]
   -            accuracy = n_correct_predictions / n_samples
   -
   -            logger.debug(f"Accuracy: {accuracy.item():.2%}")
   -            logger.debug(f"Average Loss: {loss.item()}")
   -
   -            # Advance the progress bar one step and update the progress bar text.
   +            # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
                progress_bar.update(1)
   -            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
   +            progress_bar.set_postfix(loss=loss, accuracy=accuracy)
            progress_bar.close()

   -        val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
   +        val_loss, val_accuracy = validation_loop(state, valid_dataloader)
            logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

        print("Done!")


   -@torch.no_grad()
   -def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
   -    model.eval()
   +def cross_entropy_loss(logits, labels, num_classes=10):
   +    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
   +    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
   +    loss = jnp.mean(loss)
   +    return loss

   -    total_loss = 0.0
   -    n_samples = 0
   -    correct_predictions = 0

   -    for batch in dataloader:
   -        batch = tuple(item.to(device) for item in batch)
   -        x, y = batch
   +@jax.jit
   +def train_step(state, batch, dropout_rng):
   +    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

   -        logits: Tensor = model(x)
   -        loss = F.cross_entropy(logits, y)
   +    def loss_fn(params):
   +        variables = {'params': params, 'batch_stats': state.batch_stats}
   +        logits, new_model_state = state.apply_fn(variables, batch['image'], train=True,
   +                                                 rngs={'dropout': dropout_rng}, mutable='batch_stats')
   +        loss = cross_entropy_loss(logits, batch['label'])
   +        accuracy = jnp.sum(jnp.argmax(logits, -1) == batch['label'])
   +        return loss, (accuracy, new_model_state)

   -        batch_n_samples = x.shape[0]
   -        batch_correct_predictions = logits.argmax(-1).eq(y).sum()
   +    (loss, (accuracy, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
   +    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
   +    return new_state, loss, accuracy

   -        total_loss += loss.item()
   -        n_samples += batch_n_samples
   -        correct_predictions += batch_correct_predictions

   -    accuracy = correct_predictions / n_samples
   +@jax.jit
   +def validation_step(state, batch):
   +    variables = {'params': state.params, 'batch_stats': state.batch_stats}
   +    logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
   +    loss = cross_entropy_loss(logits, batch['label'])
   +    batch_correct_predictions = jnp.sum(jnp.argmax(logits, -1) == batch['label'])
   +    return loss, batch_correct_predictions
   +
   +
   +@torch.no_grad()
   +def validation_loop(state, dataloader: DataLoader):
   +    losses = []
   +    correct_predictions = []
   +    for input, target in dataloader:
   +        batch = {
   +            'image': input,
   +            'label': target,
   +        }
   +        loss, batch_correct_predictions = validation_step(state, batch)
   +        losses.append(loss)
   +        correct_predictions.append(batch_correct_predictions)
   +
   +    total_loss = np.sum(losses)
   +    accuracy = np.mean(correct_predictions)
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
   -        root=dataset_path, transform=transforms.ToTensor(), download=True, train=True
   +        root=dataset_path, transform=ToArray(), download=True, train=True
        )
        test_dataset = CIFAR10(
   -        root=dataset_path, transform=transforms.ToTensor(), download=True, train=False
   +        root=dataset_path, transform=ToArray(), download=True, train=False
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


**model.py**

.. code:: python

   from functools import partial
   from typing import Any, Sequence

   import jax.numpy as jnp

   from flax import linen as nn


   ModuleDef = Any


   class ConvBlock(nn.Module):
       channels: int
       kernel_size: int
       norm: ModuleDef
       stride: int = 1
       act: bool = True

       @nn.compact
       def __call__(self, x):
           x = nn.Conv(self.channels, (self.kernel_size, self.kernel_size), strides=self.stride,
                       padding='SAME', use_bias=False, kernel_init=nn.initializers.kaiming_normal())(x)
           x = self.norm()(x)
           if self.act:
               x = nn.swish(x)
           return x


   class ResidualBlock(nn.Module):
       channels: int
       conv_block: ModuleDef

       @nn.compact
       def __call__(self, x):
           channels = self.channels
           conv_block = self.conv_block

           shortcut = x

           residual = conv_block(channels, 3)(x)
           residual = conv_block(channels, 3, act=False)(residual)

           if shortcut.shape != residual.shape:
               shortcut = conv_block(channels, 1, act=False)(shortcut)

           gamma = self.param('gamma', nn.initializers.zeros, 1, jnp.float32)
           out = shortcut + gamma * residual
           out = nn.swish(out)
           return out


   class Stage(nn.Module):
       channels: int
       num_blocks: int
       stride: int
       block: ModuleDef

       @nn.compact
       def __call__(self, x):
           stride = self.stride
           if stride > 1:
               x = nn.max_pool(x, (stride, stride), strides=(stride, stride))
           for _ in range(self.num_blocks):
               x = self.block(self.channels)(x)
           return x


   class Body(nn.Module):
       channel_list: Sequence[int]
       num_blocks_list: Sequence[int]
       strides: Sequence[int]
       stage: ModuleDef

       @nn.compact
       def __call__(self, x):
           for channels, num_blocks, stride in zip(self.channel_list, self.num_blocks_list, self.strides):
               x = self.stage(channels, num_blocks, stride)(x)
           return x


   class Stem(nn.Module):
       channel_list: Sequence[int]
       stride: int
       conv_block: ModuleDef

       @nn.compact
       def __call__(self, x):
           stride = self.stride
           for channels in self.channel_list:
               x = self.conv_block(channels, 3, stride=stride)(x)
               stride = 1
           return x


   class Head(nn.Module):
       classes: int
       dropout: ModuleDef

       @nn.compact
       def __call__(self, x):
           x = jnp.mean(x, axis=(1, 2))
           x = self.dropout()(x)
           x = nn.Dense(self.classes)(x)
           return x


   class ResNet(nn.Module):
       classes: int
       channel_list: Sequence[int]
       num_blocks_list: Sequence[int]
       strides: Sequence[int]
       head_p_drop: float = 0.

       @nn.compact
       def __call__(self, x, train=True):
           norm = partial(nn.BatchNorm, use_running_average=not train)
           dropout = partial(nn.Dropout, rate=self.head_p_drop, deterministic=not train)
           conv_block = partial(ConvBlock, norm=norm)
           residual_block = partial(ResidualBlock, conv_block=conv_block)
           stage = partial(Stage, block=residual_block)

           x = Stem([32, 32, 64], self.strides[0], conv_block)(x)
           x = Body(self.channel_list, self.num_blocks_list, self.strides[1:], stage)(x)
           x = Head(self.classes, dropout)(x)
           return x


**Running this example**


.. code-block:: bash

    $ sbatch job.sh
