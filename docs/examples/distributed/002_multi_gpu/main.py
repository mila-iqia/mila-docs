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
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from tqdm import tqdm


def main():
    # Check GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")

    num_workers = 4
    training_epochs = 10
    learning_rate = 5e-4
    batch_size = 512

    world_size = int(os.environ.get("WORLD_SIZE", "1"))  # SLURM_NTASKS_PER_NODE
    rank = int(os.environ.get("RANK", "0"))  # SLURM_PROCID
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
        handlers=[rich.logging.RichHandler(markup=True)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"World size: {world_size}, global rank: {rank}, local rank: {local_rank}")

    # Initialize distributed
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    is_master = torch.distributed.get_rank() == 0
    device = torch.device("cuda", local_rank)
    from torchvision import transforms

    # Obtain CIFAR10.
    dataset_path = os.environ.get("SLURM_TMPDIR", "../dataset")
    #   Only master (rank-0) downloads the dataset if needed!
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    if is_master:
        # Download THEN Barrier
        train_dataset = CIFAR10(dataset_path, transform=train_transform, download=is_master)
        valid_dataset = CIFAR10(dataset_path, transform=ToTensor(), download=is_master)
        torch.distributed.barrier()
    else:  # Barrier  THEN *NO* Download!
        torch.distributed.barrier()
        train_dataset = CIFAR10(dataset_path, transform=train_transform, download=False)
        valid_dataset = CIFAR10(dataset_path, transform=ToTensor(), download=False)

    # Batch size is now interpreted as the "global" batch size, across all GPUS.
    if batch_size % world_size != 0 and rank == world_size - 1:
        # The last GPU will get a slightly larger batch size in this case.
        local_batch_size = (batch_size // world_size) + batch_size % world_size
    else:
        local_batch_size = batch_size // world_size

    val_split = 0.1
    val_split_seed = 42
    # NOTE: Having different transformations for train and validation complicates things quite a
    # bit.
    # Make absolutely sure that all workers get the same splits.
    train_dataset, _ = random_split(
        train_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
    )
    _, valid_dataset = random_split(
        valid_dataset, ((1 - val_split), val_split), torch.Generator().manual_seed(val_split_seed)
    )
    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)

    # Create a model and move it to the GPU for this process.
    model = resnet18(num_classes=10)
    model.to(device=device)
    # Wrap the model with DistributedDataParallel
    # (See https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # NOTE: The train DataLoader is created at the start of each epoch
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    validation_dataloader = DataLoader(
        valid_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=valid_sampler,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(training_epochs):
        # NOTE: Here we need to call `set_epoch` so the ordering changes at each epoch.
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

            local_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
            # The actual local batch size (might differ from local_batch_size for the last batch)
            local_n_samples = torch.as_tensor(logits.shape[0], device=device)

            # NOTE: Creating new tensors to hold the "global" values, but this isn't required.
            n_samples = local_n_samples.clone()  # Will store the effective batch size.
            loss = local_loss.clone()  # Will store the average loss across all workers.
            correct_predictions = local_correct_predictions.clone()
            local_accuracy = local_correct_predictions / local_n_samples

            torch.distributed.all_reduce(correct_predictions, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(n_samples, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss.div_(world_size)  # Divide the loss by the number of workers.
            accuracy = correct_predictions / n_samples

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
    # NOTE: You could save


@torch.no_grad()
def validation_loop(
    model: nn.Module,
    validation_dataloader: DataLoader,
    device: torch.device,
):
    model.eval()

    epoch_val_loss = torch.as_tensor(0.0, device=device)
    epoch_n_samples = torch.as_tensor(0, device=device)
    epoch_correct_predictions = torch.as_tensor(0, device=device)

    for local_batch in validation_dataloader:
        local_batch = tuple(item.to(device) for item in local_batch)
        x, y = local_batch

        logits: Tensor = model(x)

        loss = F.cross_entropy(logits, y)
        n_samples = x.shape[0]
        correct_predictions = logits.argmax(-1).eq(y).sum()

        epoch_val_loss += loss
        epoch_n_samples += n_samples
        epoch_correct_predictions += correct_predictions

    # Sum up the metrics we gathered on each worker before returning them.
    torch.distributed.all_reduce(epoch_val_loss, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(epoch_correct_predictions, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(epoch_n_samples, op=torch.distributed.ReduceOp.SUM)

    validation_accuracy = epoch_correct_predictions / epoch_n_samples
    return epoch_val_loss, validation_accuracy


if __name__ == "__main__":
    main()
