"""Multi-GPU Training example.

Other interesting resources:
- https://sebarnold.net/dist_blog/
- https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide
"""
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
    batch_size = 512

    local_rank, rank, world_size = setup_from_open_clip()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
        handlers=[rich.logging.RichHandler(markup=True)],
    )

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = rank % torch.cuda.device_count()
    is_master = rank == 0

    num_workers = get_num_workers()
    logger = logging.getLogger(__name__)
    logger.info(f"World size: {world_size}, global rank: {rank}, local rank: {local_rank}")

    device = torch.device("cuda", local_rank)

    dataset_path = os.environ.get("SLURM_TMPDIR", "../dataset")

    # Create a model and move it to the GPU for this process.
    model = resnet18(num_classes=10)
    model.to(device=device)

    # Wrap the model with DistributedDataParallel
    # (See https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # Batch size is now interpreted as the "global" batch size, across all GPUS.
    if batch_size % world_size != 0 and rank == world_size - 1:
        # The last GPU will get a slightly larger batch size in this case.
        local_batch_size = (batch_size // world_size) + batch_size % world_size
    else:
        local_batch_size = batch_size // world_size

    train_dataset, valid_dataset, test_dataset = make_datasets(dataset_path, is_master=is_master)
    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)
    test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=False,  # shuffling is done in the sampler, not the dataloader.
        num_workers=num_workers,
        sampler=train_sampler,
    )
    validation_dataloader = DataLoader(
        valid_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=valid_sampler,
    )
    test_dataloader = DataLoader(  # note: unused in this example here.
        test_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(training_epochs):
        # NOTE: Here we need to call `set_epoch` so the ordering is able to change at each epoch.
        train_sampler.set_epoch(epoch)

        progress_bar = tqdm(
            total=len(train_dataloader), desc=f"Training epoch {epoch}", disable=not is_master
        )

        model.train()
        # Training loop
        for local_batch in train_dataloader:
            # Move the batch to the GPU before we pass it to the model
            local_batch = tuple(item.to(device) for item in local_batch)
            x, y = local_batch
            # (distributed) Forward pass
            logits: Tensor = model(x)

            # Calculate the loss for this particular batch of data:
            local_loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            local_loss.backward()
            # NOTE: nn.DistributedDataParallel automatically averages the gradients across devices.
            optimizer.step()

            # "local" metrics: calculated with the results for this worker
            local_n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
            # The actual local batch size (might differ from local_batch_size for the last batch)
            local_n_samples = logits.shape[0]
            local_accuracy = local_n_correct_predictions / local_n_samples

            # "global" metrics: calculated with the results from all workers
            # NOTE: Creating new tensors to hold the "global" values, but this isn't required.
            n_correct_predictions = local_n_correct_predictions.clone()
            # Actual (global) batch size for this step.
            n_samples = torch.as_tensor(local_n_samples, device=device)
            # Will store the average loss across all workers.
            loss = local_loss.clone()

            torch.distributed.all_reduce(n_correct_predictions, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(n_samples, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss.div_(world_size)  # Report the average loss across all workers.
            accuracy = n_correct_predictions / n_samples

            logger.debug(f"(local) accuracy: {local_accuracy:.2%}")
            logger.debug(f"(local) loss: {local_loss.item()}")
            # NOTE: This would log the same values in all workers. Only logging on master:
            if is_master:
                logger.debug(f"Accuracy: {accuracy.item():.2%}")
                logger.debug(f"Average Loss: {loss.item()}")

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
        progress_bar.close()

        val_loss, val_accuracy = validation_loop(model, validation_dataloader, device)
        # NOTE: This would log the same values in all workers. Only logging on master:
        if is_master:
            logger.info(
                f"Epoch {epoch}: Validation loss: {val_loss:.3f} Val accuracy: {val_accuracy:.2%}"
            )

    print("Done!")


@torch.no_grad()
def validation_loop(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    model.eval()

    val_loss = torch.as_tensor(0.0, device=device)
    n_samples = torch.as_tensor(0, device=device)
    correct_predictions = torch.as_tensor(0, device=device)

    for local_batch in dataloader:
        local_batch = tuple(item.to(device) for item in local_batch)
        x, y = local_batch

        logits: Tensor = model(x)

        loss = F.cross_entropy(logits, y)
        batch_n_samples = x.shape[0]
        batch_correct_predictions = logits.argmax(-1).eq(y).sum()

        val_loss += loss
        n_samples += batch_n_samples
        correct_predictions += batch_correct_predictions

    # Sum up the metrics we gathered on each worker before returning the overall val metrics.
    torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(correct_predictions, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(n_samples, op=torch.distributed.ReduceOp.SUM)

    validation_accuracy = correct_predictions / n_samples
    return val_loss, validation_accuracy


def setup():
    if "WORLD_SIZE" in os.environ:
        # Job launched with `torchrun`:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    else:
        # Job launched with `srun`
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])

    # Initialize distributed
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
        handlers=[rich.logging.RichHandler(markup=True)],
    )


def setup_from_open_clip():
    # Check GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")

    if "SLURM_PROCID" in os.environ:
        # DDP Job is being run via `srun` on a slurm cluster.
        local_rank = int(os.environ["SLURM_LOCALID"])
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        # SLURM var -> torch.distributed vars in case needed
        # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
        # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
        os.environ["LOCAL_RANK"] = str(local_rank)
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
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    return local_rank, rank, world_size


def make_datasets(
    dataset_path: str,
    is_master: bool,  # Added: Only master (rank-0) downloads the dataset if needed!
    val_split=0.1,
    val_split_seed=42,
):
    # Obtain CIFAR10.
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    # Only master (rank-0) downloads the dataset if necessary.
    if is_master:
        # Download (if necessary) THEN Barrier
        train_dataset = CIFAR10(dataset_path, transform=train_transform, download=True)
        valid_dataset = CIFAR10(dataset_path, transform=val_transform, download=True)
        test_dataset = CIFAR10(dataset_path, transform=val_transform, download=True, train=False)
        torch.distributed.barrier()
    else:
        # Barrier  THEN *NO* Download!
        torch.distributed.barrier()
        train_dataset = CIFAR10(dataset_path, transform=train_transform, download=False)
        valid_dataset = CIFAR10(dataset_path, transform=val_transform, download=False)
        test_dataset = CIFAR10(dataset_path, transform=val_transform, train=False, download=False)

    # NOTE: Having different transformations for train and validation complicates things quite a
    # bit.
    # Make absolutely sure that all workers get the same splits.
    train_dataset, _ = random_split(
        train_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
    )
    _, valid_dataset = random_split(
        valid_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
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
