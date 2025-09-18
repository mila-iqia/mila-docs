"""ImageNet Distributed training script.

# Features:
- Multi-GPU / Multi-node training with DDP
- Wandb logging
- Checkpointing
- Profiling with the PyTorch profiler and tensorboard

# Potential Improvements - to be added as an exercise! 😉
- Use Automatic Mixed Precision (AMP) to take advantage of the hardware
- Add code checkpointing with git to avoid unexpected bugs
- Use a larger model that doesn't fit inside a single GPU with FSDP.
"""

import dataclasses
import datetime
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import numpy as np
import rich.logging
import rich.pretty
import simple_parsing
import sklearn
import sklearn.model_selection
import torch
import torchvision
import tqdm
import tqdm.rich
import wandb
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.nn import functional as F
from torch.profiler import profile, tensorboard_trace_handler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageNet
from torchvision.transforms import v2 as transforms

JOB_ID = os.environ["SLURM_JOB_ID"]  # you absolutely need to be within a slurm job!
SCRATCH = Path(os.environ["SCRATCH"])
SLURM_TMPDIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
assert SLURM_TMPDIR.exists(), f"SLURM_TMPDIR (assumed {SLURM_TMPDIR}) should exist!"

# Set any missing environment variables so that `torch.distributed.init_process_group`
# works properly, namely RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, (LOCAL_RANK).
#
# The accompanying sbatch script already does this in bash, which is preferable, since
# you need to make sure that these environment variables are set before any torch operations
# are executed. (Some modules might inadvertently initialize cuda when imported which is a problem).
#
# Also doing this here just in case you're using a different sbatch script or running this from
# the vscode terminal or with the vscode debugger.
# Using the Vscode debugger to debug multi-gpu jobs is very convenient.
# When debugging in a vscode window created by `mila code`, we do not have the slurm
# environment variables (except SLURM_JOB_ID), but have the torchrun ones.

# Note: here by using .setdefault we don't overwrite env variables that are already set,
# so you could in principle use this in a workflow based on srun + torchrun or
# srun + 'accelerate launch'.
#
# If neither the SLURM nor the torch distributed env vars are set, raise an error.
if "SLURM_PROCID" not in os.environ and "RANK" not in os.environ:
    raise RuntimeError(
        "Both the SLURM and the torch distributed env vars are not set! "
        "This indicates that you might be running this script in something like the "
        "vscode terminal with `python <this_file>`.\n"
        f"Consider relaunching the same command with srun instead, like so: \n"
        f"➡️ srun --pty {sys.executable} {' '.join(sys.argv)}\n"
        "See https://slurm.schedmd.com/srun.html for more info."
    )

# This will raise an error if both are unset. This is desired.
RANK = int(os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "")))
LOCAL_RANK = int(os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "")))
WORLD_SIZE = int(os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "")))
MASTER_PORT = int(os.environ.setdefault("MASTER_PORT", str(10000 + int(JOB_ID) % 10000)))
if "SLURM_JOB_NODELIST" in os.environ:
    # Get the hostname of the first node, for example: "cn-l[084-085]" --> cn-l084
    _first_node = subprocess.check_output(
        f"scontrol show hostnames {os.environ['SLURM_JOB_NODELIST']}", text=True, shell=True
    ).split()[0]
    MASTER_ADDR = os.environ.setdefault("MASTER_ADDR", _first_node)
else:
    MASTER_ADDR = os.environ.setdefault("MASTER_ADDR", "127.0.0.1")


class DummyModel(nn.Module):
    """Dummy model used while debugging - uses almost no compute or memory.

    Examples of when this is useful:
    -   to check if data loading is the bottleneck, we can pull samples from the dataloader
        as fast as possible and compare that throughput (in samples/second) to the same
        during training. If the two are similar, then the dataloader is the bottleneck.
        Using a dummy model like this makes it so we don't have to modify our training loop
        to do this kind of sanity check.
    """

    def __init__(self, num_classes: int, **_kwargs):
        super().__init__()
        self.num_classes = num_classes
        # A dummy weight..
        self.linear = nn.Linear(1, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.flatten(1).mean(1, keepdim=True))


models: dict[str, Callable[..., nn.Module]] = {
    "debug_model": DummyModel,
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
    "vit_b_16": torchvision.models.vit_b_16,
    "vit_b_32": torchvision.models.vit_b_32,
    "vit_l_16": torchvision.models.vit_l_16,
    "vit_l_32": torchvision.models.vit_l_32,
}


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=f"[{RANK + 1}/{WORLD_SIZE}] %(name)s - %(message)s ",
    handlers=[rich.logging.RichHandler(markup=True)],
    force=True,
)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    """Dataclass that contains the command-line arguments for this script."""

    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128

    pretrained: bool = False
    """Whether to use a pretrained model or start from a random initialization."""

    checkpoint_dir: Path = SCRATCH / "checkpoints" / JOB_ID
    """Where checkpoints are stored."""

    dataset_path: Path = SLURM_TMPDIR / "data"
    """Where to look for the dataset."""

    use_fake_data: bool = False
    """If true, use torchvision.datasets.FakeData instead of ImageNet.

    Useful for debugging.
    """

    num_workers: int = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))
    """Number of dataloader workers."""

    seed: int = 42
    """Base random seed for everything except the train/validation split."""

    val_seed: int = 0
    """Random seed used to create the train/validation split."""

    model_name: str = simple_parsing.choice(*models.keys(), default="resnet18")
    """Which model function to use."""

    compile: bool = False
    """If true, use torch.compile to compile the model."""

    verbose: int = simple_parsing.field(alias="-v", action="count", default=0)
    """Increase logging verbosity (can be specified multiple times)."""

    logging_interval: int = 10
    """Interval (in batches) between logging training metrics."""

    use_amp: bool = False
    """If True, use automatic mixed precision (AMP) for training."""

    wandb_run_name: str = JOB_ID
    """Name for the wandb run."""

    wandb_run_id: str = JOB_ID
    """Unique ID for the Weights & Biases run.

    Used to resume a run if the job is restarted.
    """

    wandb_group: str | None = None

    wandb_project: str = "codingtips_profiling_example"


def main():
    # Use an argument parser so we can pass hyperparameters from the command line.
    # You can use plain argparse if you like. Simple-parsing is an extension of argparse for dataclasses.
    args: Args = simple_parsing.parse(
        Args,
        # Arguments can be passed with either --arg_name or --arg-name
        add_option_string_dash_variants=simple_parsing.DashVariant.UNDERSCORE_AND_DASH,
    )

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    assert torch.distributed.is_available()
    # https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html#constructing-the-process-group
    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        rank=RANK,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(minutes=5),
    )
    is_master = RANK == 0
    _is_local_master = LOCAL_RANK == 0

    device = torch.device("cuda", LOCAL_RANK)

    print(f"Using random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.setLevel(
        logging.WARNING
        if args.verbose == 0
        else logging.INFO
        if args.verbose == 1
        else logging.DEBUG
    )
    logger.info(f"World size: {WORLD_SIZE}, global rank: {RANK}, local rank: {LOCAL_RANK}")
    if is_master:
        logger.info("Args: ")
        rich.pretty.pprint(dataclasses.asdict(args))

    # Create a model and move it to the GPU.

    kwargs = {} if not args.pretrained else {"weights": "DEFAULT"}
    model = models[args.model_name](num_classes=1000, **kwargs)
    model.to(device=device)
    # https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html#multi-gpu-training-with-ddp
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.compile:
        # TODO: do this before or after the DDP wrapper?
        model = torch.compile(model)
    # Wrap the model with DistributedDataParallel
    # (See https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Setup the dataset.
    train_dataset, valid_dataset, test_dataset = make_datasets(
        args.dataset_path,
        val_split_seed=args.val_seed,
        use_fake_data=args.use_fake_data,
    )

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(
        dataset=train_dataset, shuffle=True, num_replicas=WORLD_SIZE, rank=RANK, seed=args.seed
    )
    valid_sampler = DistributedSampler(
        dataset=valid_dataset, shuffle=False, num_replicas=WORLD_SIZE, rank=RANK
    )
    test_sampler = DistributedSampler(
        dataset=test_dataset, shuffle=False, num_replicas=WORLD_SIZE, rank=RANK
    )
    # TODO: make sure that the dataloader state is restored properly.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=valid_sampler,
        pin_memory=True,
    )
    _test_dataloader = DataLoader(  # NOTE: Not used in this example.
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=test_sampler,
        pin_memory=True,
    )
    global_batch_size = args.batch_size * WORLD_SIZE
    logger.info(f"Global batch size: {global_batch_size}")

    # Load the latest checkpoint if it exists.
    if previous_checkpoints := list(args.checkpoint_dir.glob("*.pt")):
        # Checkpoints are named like `epoch_0.pt`, `epoch_1.pt`. Find the latest.
        latest_checkpoint = max(previous_checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        _checkpoint_epoch, step, num_samples = load_checkpoint(
            latest_checkpoint, model=model, optimizer=optimizer, device=device
        )
        starting_epoch = _checkpoint_epoch + 1
        total_updates = step
        total_num_samples = num_samples
        logger.debug(
            f"Starting training from epoch {starting_epoch} (step {step}, {total_num_samples} total samples)"
        )
    else:
        starting_epoch = 0
        total_updates = 0
        total_num_samples = 0
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Starting training from scratch")

    # Initialize wandb logging.
    # Normally you would only do this in the first task (rank 0), but here we do it in all tasks
    # using the new "shared" feature of wandb. This makes it much easier to track the GPU util of
    # all gpus on all nodes in the job.
    # See this link for more info:
    # - https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        config=dataclasses.asdict(args)
        | {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        group=args.wandb_group,
        # Resume an existing run with the same ID if the job is restarting after being preempted.
        resume=(
            "must"  # 'must' will ignore all logged data until the previous step is reached.
            if (int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0) or previous_checkpoints
            else "allow"  # will log new data in the same run, which makes weird jagged plots.
        ),
        # NOTE: Would be *really* nice to use this resume feature, but this is new
        # at the time of writing (2025-09) and needs to be enabled for your project
        # by contacting wandb support.
        # resume_from=f"{JOB_ID}?_step={total_updates}",
        # Use the new "shared" mode to log system utilization metrics from all tasks in the job:
        settings=wandb.Settings(
            mode="shared",
            x_primary=is_master,
            x_label=f"task_{RANK}",
            x_stats_gpu_device_ids=[LOCAL_RANK],
            x_update_finish_state=not is_master,
        ),
    )
    # Specify the step metric (x-axis) and the metric to log against it (y-axis)
    run.define_metric("train/*", step_metric="updates")
    run.define_metric("valid/*", step_metric="epoch")

    # Create the PyTorch profiler with a schedule that will output some traces that can be inspected with tensorboard.
    # https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs
    # To view the traces, run `uvx tensorboard --with=torch_tb_profiler --logdir checkpoints`
    profiler = profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=2, repeat=1),
        on_trace_ready=tensorboard_trace_handler(
            str(args.checkpoint_dir), worker_name=f"rank_{RANK}"
        ),
    )

    ###################
    ## Training loop ##
    ###################

    for epoch in range(starting_epoch, args.epochs):
        logger.debug(f"Starting epoch {epoch}/{args.epochs}")
        # Important so each epoch uses a different ordering for the training samples.
        train_sampler.set_epoch(epoch)

        model.train()

        # Using a progress bar when in an interactive terminal. It also shows the throughput in samples/second.
        # If we're going to enable verbose logging within an epoch (for example to help identify issues),
        # it makes sense to use the progress bar from rich so that the logs are displayed nicely.
        # However, it doesn't support the `unit_scale` and `unit` arguments atm so we disable those arguments.
        pbar_type = tqdm.rich.tqdm_rich if args.verbose >= 2 else tqdm.tqdm
        assert isinstance(train_dataloader.batch_size, int)
        progress_bar = pbar_type(
            train_dataloader,
            desc=f"Train epoch {epoch}/{args.epochs - 1}",
            # Don't use a progress bar if outputting to a slurm output file or when not in task 0
            disable=(not sys.stdout.isatty() or not is_master),
            unit_scale=False if pbar_type is tqdm.rich.tqdm_rich else global_batch_size,
            unit="batches" if pbar_type is tqdm.rich.tqdm_rich else "samples",
            dynamic_ncols=True,  # allow window resizing
        )

        t = time.perf_counter()
        for batch_index, batch in enumerate(
            # We only create the profiling traces in the first epoch.
            profile_loop(progress_bar, profiler) if epoch == 0 else progress_bar
        ):
            # Move the batch to the GPU before we pass it to the model
            batch = tuple(item.to(device) for item in batch)
            x, y = batch

            loss, accuracy, n_samples = training_step(model, x, y, optimizer, is_master=is_master)

            total_updates += 1
            total_num_samples += n_samples

            # Simple training speed calculation in samples/sec using the global batch size.
            new_t = time.perf_counter()
            dt = new_t - t
            samples_per_sec = n_samples / dt
            t = new_t

            if is_master and (batch_index + 1) % args.logging_interval == 0:
                # update the progress bar text.
                _loss = loss.item()
                _accuracy = accuracy.item()
                progress_bar.set_postfix(
                    loss=f"{_loss:.3f}",
                    accuracy=f"{_accuracy:.2%}",
                )
                # TODO: Could be interesting to also log the local loss / accuracy values on all workers.
                wandb.log(
                    {
                        "train/loss": _loss,
                        "train/accuracy": _accuracy,
                        "train/samples_per_sec": samples_per_sec,
                        "epoch": epoch,
                        "updates": total_updates,
                        "samples": total_num_samples,
                    }
                )
        progress_bar.close()

        t = time.perf_counter()
        val_loss, val_accuracy, val_samples = validation_loop(model, valid_dataloader, device)
        dt = time.perf_counter() - t
        val_sps = val_samples / dt
        logger.info(
            f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%} samples/sec: {val_sps:.1f}"
        )
        wandb.log(
            {
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,
                "val/samples_per_sec": val_sps,
                "epoch": epoch,
            }
        )

        # Only save the checkpoint from the master process.
        # TODO: Make sure this doesn't cause a timeout if it takes too long.
        if is_master:
            save_checkpoint(
                checkpoint_path=args.checkpoint_dir / f"epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                step=total_updates,
                num_samples=int(total_num_samples),
            )

    torch.distributed.destroy_process_group()
    print("Done!")


def training_step(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    optimizer: torch.optim.Optimizer,
    is_master: bool = False,
):
    # Forward pass
    logits: Tensor = model(x)

    local_loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    # NOTE: nn.DistributedDataParallel automatically averages the gradients across devices.
    local_loss.backward()
    optimizer.step()

    # Calculate some metrics:

    # TODO: Use torchmetrics instead of calculating metrics ourselves? (But then
    # we don't see (and learn) how to use the communication primitives!)

    # local metrics
    local_n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
    local_n_samples = logits.shape[0] * torch.ones(1, device=local_loss.device, dtype=torch.int32)
    local_accuracy = local_n_correct_predictions / local_n_samples

    # "global" metrics: calculated with the results from all workers
    # Creating new tensors to hold the "global" values, but this isn't required.
    # Reduce the local metrics across all workers, sending the result to rank 0.

    n_correct_predictions = local_n_correct_predictions.clone()
    n_samples = local_n_samples.clone()
    loss = local_loss.clone()

    torch.distributed.reduce(loss, dst=0, op=ReduceOp.AVG)
    # Summing n_correct and n_samples to get accuracy is resilient to
    # workers having different number of samples.
    # This could happen if the number of batches is not divisible by the number of batches
    # and if the distributed sampler is not set to drop the last incomplete batch.
    torch.distributed.reduce(n_correct_predictions, dst=0, op=ReduceOp.SUM)
    torch.distributed.reduce(n_samples, dst=0, op=ReduceOp.SUM)
    accuracy = n_correct_predictions / n_samples

    # FIXME: The .item calls here happen even if we don't even want to show these values!
    if WORLD_SIZE > 1:
        logger.debug(f"(local) Loss: {local_loss.item():.2f} Accuracy: {local_accuracy.item():.2%}")
    if is_master:  # Otherwise this would log the same values once per worker.
        logger.debug(
            ("Average" if WORLD_SIZE > 1 else "")
            + f"Loss: {loss.item():.2f} Accuracy: {accuracy.item():.2%}"
        )
    return loss, accuracy, n_samples


@torch.no_grad()
def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()

    epoch_loss = torch.zeros(1, device=device)
    num_samples = torch.zeros(1, device=device, dtype=torch.int32)
    correct_predictions = torch.zeros(1, device=device, dtype=torch.int32)
    assert isinstance(dataloader.batch_size, int)

    progress_bar = tqdm.tqdm(
        dataloader,
        desc="Validation",
        unit_scale=dataloader.batch_size * WORLD_SIZE,
        unit="samples",
        # Don't use a progress bar if outputting to a slurm output file or when not in task 0
        disable=(not sys.stdout.isatty() or RANK != 0),
    )
    # NOTE: Because of DDP and distributed sampler, the last batch might have repeated samples,
    # leading to slightly imprecise metrics.
    for batch in progress_bar:
        batch = tuple(item.to(device) for item in batch)
        x, y = batch

        logits: Tensor = model(x)
        loss = F.cross_entropy(logits, y)

        batch_n_samples = x.shape[0]
        batch_correct_predictions = logits.argmax(-1).eq(y).sum()

        epoch_loss += loss
        num_samples += batch_n_samples
        correct_predictions += batch_correct_predictions
    # NOTE: Here we only reduce after iteration over the entire dataset, which is more efficient
    # but wouldn't work if the model is too large to fit on a single GPU.
    torch.distributed.reduce(epoch_loss, dst=0, op=ReduceOp.SUM)
    torch.distributed.reduce(num_samples, dst=0, op=ReduceOp.SUM)
    torch.distributed.reduce(correct_predictions, dst=0, op=ReduceOp.SUM)
    epoch_average_loss = epoch_loss / num_samples
    accuracy = correct_predictions / num_samples
    return epoch_average_loss.item(), accuracy.item(), num_samples.item()


T = TypeVar("T")


def profile_loop(dataloader: Iterable[T], profiler: torch.profiler.profile) -> Iterable[T]:
    """Wraps the dataloader (or progress bar) and calls .step after each batch.

    Note, this doesn't need to be done at every epoch. It creates files used by tensorboard.
    """
    with profiler as prof:
        for batch in dataloader:
            yield batch
            prof.step()


def make_datasets(
    path: Path,
    val_split: float = 0.1,
    val_split_seed: int = 42,
    use_fake_data: bool = False,
):
    """Returns the training, validation, and test splits."""
    if use_fake_data:
        train_dataset = torchvision.datasets.FakeData(
            size=1_281_167,
            image_size=(3, 224, 224),
            num_classes=1000,
            transform=transforms.ToTensor(),
        )
        valid_dataset = torchvision.datasets.FakeData(
            size=20_000,
            image_size=(3, 224, 224),
            num_classes=1000,
            transform=transforms.ToTensor(),
        )

        test_dataset = torchvision.datasets.FakeData(
            size=50_000,
            image_size=(3, 224, 224),
            num_classes=1000,
            transform=transforms.ToTensor(),
        )
        return train_dataset, valid_dataset, test_dataset
    # TODO: Check if we put the transforms on the GPU and see if it helps performance a bit.
    train_transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ).cuda()
    test_transforms = torch.nn.Sequential(
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    train_dataset = ImageNet(root=path, transform=train_transforms, split="train")
    valid_dataset = ImageNet(root=path, transform=test_transforms, split="train")
    test_dataset = ImageNet(root=path, transform=test_transforms, split="val")

    # TODO: Add an option to limit the number of total samples in the training dataset,
    # to make it easy to check whether a randomly initialized model can overfit to a few batches.
    # if limit_num_samples:
    #     train_dataset = torch.utils.data.Subset(
    #         train_dataset, list(range(limit_num_samples))
    #     )
    #     valid_dataset = torch.utils.data.Subset(
    #         valid_dataset, list(range(limit_num_samples))
    #     )
    #     test_dataset = torch.utils.data.Subset(
    #         test_dataset, list(range(limit_num_samples))
    #     )

    # Split the training dataset into a training and validation set, based on a stratified split.
    # This is important to have a balanced distribution of classes in both sets.
    # See the sklearn.model_selection.train_test_split documentation for more info.
    n_samples = len(train_dataset)
    n_valid = int(val_split * n_samples)
    n_train = n_samples - n_valid
    train_indices, val_indices = sklearn.model_selection.train_test_split(
        np.arange(n_samples),
        train_size=n_train,
        test_size=n_valid,
        random_state=np.random.RandomState(val_split_seed),
        shuffle=True,
        stratify=train_dataset.targets,
    )
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, val_indices)

    return train_dataset, valid_dataset, test_dataset


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, int]:
    logger.info(f"Loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    nsamples = checkpoint["num_samples"]
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    cpu_rng_state = checkpoint["torch_rng_state_cpu"]
    torch.random.set_rng_state(cpu_rng_state.cpu())
    torch.cuda.random.set_rng_state_all([t.cpu() for t in checkpoint["torch_rng_state_gpu"]])
    return epoch, step, nsamples


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    step: int,
    num_samples: int,
):
    logger.info(f"Saving checkpoint at {checkpoint_path}")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "num_samples": num_samples,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state_cpu": torch.random.get_rng_state(),
        "torch_rng_state_gpu": torch.cuda.random.get_rng_state_all(),
    }
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()
