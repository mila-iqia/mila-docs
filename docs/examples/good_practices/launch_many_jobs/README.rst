.. NOTE: This file is auto-generated from examples/good_practices/launch_many_jobs/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

.. _launch_many_jobs:

Launch many jobs from same shell script
=======================================

Sometimes you may want to run the same job with different arguments. For example, you may want to launch an experiment using a few different values for a given parameter.

The naive way to do this would be to create multiple sbatch scripts, each with a different value for that parameter.
Another might be to use a single sbatch script with multiple lines, each with a different parameter value, and to then uncomment a given line before submitting the job, then commenting and uncommenting a different line before submitting another job, etc.

This example shows a  practical solution to this problem, allowing you to parameterize a job's sbatch script, and pass different values directly from the command-line when submitting the job.

In this example, our job script is a slightly modified version of the Python script from the single-GPU example, with a bit of code added so that it is able to take in values from the command-line.
The sbatch script uses the ``$@`` bash directive to pass the command-line arguments to the python script. This makes it very easy to submit multiple jobs, each with different values!

The next examples will then build on top of this one to illustrate good practices related to launching lots of jobs for hyper-parameter sweeps:

* Using SLURM Job Arrays for Hyper-Parameter Sweeps (coming soon!)
* :ref:`Running more effective Hyper-Parameter Sweeps with Orion <hpo_with_orion>`


**Prerequisites**
Make sure to read the following sections of the documentation before using this
example:

* `examples/frameworks/pytorch_setup <https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_

The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/launch_many_jobs>`_

**job.sh**

.. code:: diff

    # distributed/single_gpu/job.sh -> good_practices/launch_many_jobs/job.sh
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
    module load cuda/11.7

    # Creating the environment for the first time:
    # conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
    #     pytorch-cuda=11.7 -c pytorch -c nvidia
    # Other conda packages:
    # conda install -y -n pytorch -c conda-forge rich tqdm

    # Activate pre-existing environment.
    conda activate pytorch


    # Stage dataset into $SLURM_TMPDIR
    mkdir -p $SLURM_TMPDIR/data
    cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
    # General-purpose alternatives combining copy and unpack:
    #     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
    #     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/


    # Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
    unset CUDA_VISIBLE_DEVICES

   -# Execute Python script
   -python main.py
   +# Call main.py with all arguments passed to this script.
   +# This allows you to call the script many times with different arguments.
   +# Quotes around $@ prevent splitting of arguments that contain spaces.
   +python main.py "$@"


**main.py**

.. code:: diff

    # distributed/single_gpu/main.py -> good_practices/launch_many_jobs/main.py
    """Single-GPU training example."""
   +import argparse
    import logging
    import os
    from pathlib import Path

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
   -    training_epochs = 10
   -    learning_rate = 5e-4
   -    weight_decay = 1e-4
   -    batch_size = 128
   +    # Add an argument parser so that we can pass hyperparameters from the command line.
   +    parser = argparse.ArgumentParser(description=__doc__)
   +    parser.add_argument("--epochs", type=int, default=10)
   +    parser.add_argument("--learning-rate", type=float, default=5e-4)
   +    parser.add_argument("--weight-decay", type=float, default=1e-4)
   +    parser.add_argument("--batch-size", type=int, default=128)
   +    args = parser.parse_args()
   +
   +    training_epochs = args.epochs
   +    learning_rate = args.learning_rate
   +    weight_decay = args.weight_decay
   +    batch_size = args.batch_size

        # Check that the GPU is available
        assert torch.cuda.is_available() and torch.cuda.device_count() > 0
        device = torch.device("cuda", 0)

        # Setup logging (optional, but much better than using print statements)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
        )

        logger = logging.getLogger(__name__)
   +    logger.info(f"Arguments: {args}")

        # Create a model and move it to the GPU.
        model = resnet18(num_classes=10)
        model.to(device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

        for epoch in range(training_epochs):
            logger.debug(f"Starting epoch {epoch}/{training_epochs}")

            # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
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

                # Advance the progress bar one step and update the progress bar text.
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
            progress_bar.close()

            val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
            logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

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
You can then launch many jobs using same script with various args.

.. code-block:: bash

    $ sbatch job.sh --learning-rate 0.1
    $ sbatch job.sh --learning-rate 0.5
    $ sbatch job.sh --weight-decay 1e-3
