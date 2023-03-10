"""Multi-GPU Training example."""
import logging
import os

import rich.logging
import torch
import torch.distributed
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm


def main():
    training_epochs = 10
    learning_rate = 5e-4
    weight_decay = 1e-4
    batch_size = 128  # NOTE: This is the "local" batch size, per-GPU.

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    rank, world_size = setup()
    is_master = rank == 0
    device = torch.device("cuda", rank)

    # Setup logging (optional, but much better than using print statements)
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
        handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
    )

    logger = logging.getLogger(__name__)
    logger.info(f"World size: {world_size}, global rank: {rank}")

    # Create a model and move it to the GPU.
    model = resnet18(num_classes=10)
    model.to(device=device)

    # Wrap the model with DistributedDataParallel
    # (See https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Setup CIFAR10
    num_workers = get_num_workers()
    dataset_path = os.environ.get("SLURM_TMPDIR", "../dataset")
    train_dataset, valid_dataset, test_dataset = make_datasets(dataset_path, is_master=is_master)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)
    test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)

    # NOTE: Here `batch_size` is still the "local" (per-gpu) batch size.
    # This way, the effective batch size scales directly with number of GPUs, no need to specify it
    # in advance. You might want to adjust the learning rate and other hyper-parameters though.
    if is_master:
        logger.info(f"Effective batch size: {batch_size * world_size}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # shuffling is now done in the sampler, not the dataloader.
        sampler=train_sampler,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=valid_sampler,
    )
    test_dataloader = DataLoader(  # NOTE: Not used in this example.
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=test_sampler,
    )

    # Checkout the "checkpointing and preemption" example for more info!
    logger.debug("Starting training from scratch.")

    for epoch in range(training_epochs):
        logger.debug(f"Starting epoch {epoch}/{training_epochs}")

        # NOTE: Here we need to call `set_epoch` so the ordering is able to change at each epoch.
        train_sampler.set_epoch(epoch)

        # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
        model.train()

        # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Train epoch {epoch}",
            disable=not is_master,
        )

        # Training loop
        for batch in train_dataloader:
            # Move the batch to the GPU before we pass it to the model
            batch = tuple(item.to(device) for item in batch)
            x, y = batch

            # Forward pass
            logits: Tensor = model(x)

            local_loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            local_loss.backward()
            # NOTE: nn.DistributedDataParallel automatically averages the gradients across devices.
            optimizer.step()

            # Calculate some metrics:
            # local metrics
            local_n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
            local_n_samples = logits.shape[0]
            local_accuracy = local_n_correct_predictions / local_n_samples

            # "global" metrics: calculated with the results from all workers
            # NOTE: Creating new tensors to hold the "global" values, but this isn't required.
            n_correct_predictions = local_n_correct_predictions.clone()
            torch.distributed.all_reduce(n_correct_predictions, op=torch.distributed.ReduceOp.SUM)
            # Actual (global) batch size for this step.
            n_samples = torch.as_tensor(local_n_samples, device=device)
            torch.distributed.all_reduce(n_samples, op=torch.distributed.ReduceOp.SUM)
            # Will store the average loss across all workers.
            loss = local_loss.clone()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss.div_(world_size)  # Report the average loss across all workers.

            accuracy = n_correct_predictions / n_samples

            logger.debug(f"(local) Accuracy: {local_accuracy:.2%}")
            logger.debug(f"(local) Loss: {local_loss.item()}")
            # NOTE: This would log the same values in all workers. Only logging on master:
            if is_master:
                logger.debug(f"Accuracy: {accuracy.item():.2%}")
                logger.debug(f"Average Loss: {loss.item()}")

            # Advance the progress bar one step, and update the "postfix" () the progress bar. (nicer than just)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
        progress_bar.close()

        val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
        # NOTE: This would log the same values in all workers. Only logging on master:
        if is_master:
            logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

    print("Done!")


@torch.no_grad()
def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()

    total_loss = torch.as_tensor(0.0, device=device)
    n_samples = torch.as_tensor(0, device=device)
    correct_predictions = torch.as_tensor(0, device=device)

    for batch in dataloader:
        batch = tuple(item.to(device) for item in batch)
        x, y = batch

        logits: Tensor = model(x)
        loss = F.cross_entropy(logits, y)

        batch_n_samples = x.shape[0]
        batch_correct_predictions = logits.argmax(-1).eq(y).sum()

        total_loss += loss
        n_samples += batch_n_samples
        correct_predictions += batch_correct_predictions

    # Sum up the metrics we gathered on each worker before returning the overall val metrics.
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(correct_predictions, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(n_samples, op=torch.distributed.ReduceOp.SUM)

    accuracy = correct_predictions / n_samples
    return total_loss, accuracy


def setup():
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")

    if "SLURM_PROCID" in os.environ:
        # DDP Job is being run via `srun` on a slurm cluster.
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        # SLURM var -> torch.distributed vars in case needed
        # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
        # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    else:
        # DDP via torchrun, torch.distributed.launch
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    return rank, world_size


def make_datasets(
    dataset_path: str,
    is_master: bool,
    val_split: float = 0.1,
    val_split_seed: int = 42,
):
    """Returns the training, validation, and test splits for CIFAR10.

    NOTE: We don't use image transforms here for simplicity.
    Having different transformations for train and validation would complicate things a bit.
    Later examples will show how to do the train/val/test split properly when using transforms.

    NOTE: Only the master process (rank-0) downloads the dataset if necessary.
    """
    # - Master: Download (if necessary) THEN Barrier
    # - others: Barrier THEN *NO* Download
    if not is_master:
        # Wait for the master process to finish downloading (reach the barrier below)
        torch.distributed.barrier()
    train_dataset = CIFAR10(
        root=dataset_path, transform=transforms.ToTensor(), download=is_master, train=True
    )
    test_dataset = CIFAR10(
        root=dataset_path, transform=transforms.ToTensor(), download=is_master, train=False
    )
    if is_master:
        # Join the workers waiting in the barrier above. They can now load the datasets from disk.
        torch.distributed.barrier()
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


if __name__ == "__main__":
    main()
