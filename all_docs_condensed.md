# Mila Technical Documentation — Condensed Reference for ML Students

> A focused reference distilled from the full Mila docs for Masters/PhD students.
> Covers: cluster access, SLURM, storage, GPU/partition choice, Python envs with `uv`,
> experiment tracking, distributed training, checkpointing, HPO, containers, and FAQ.

---

## 1. Cluster access

### Account and onboarding

1. Ask your supervisor for a Mila invite → obtain `@mila.quebec` account.
2. Complete the IT Onboarding Training (on the Mila intranet) and submit the quiz.
   IT then emails cluster connection details (username, etc.) within ~48h.
3. IT activates Multi-Factor Authentication (MFA) and sends a one-time **registration token**.

Windows users: install **WSL2** (`wsl --install Ubuntu`) and run everything inside Ubuntu.

### MFA setup

Cluster access requires **two factors**: your SSH key + a second factor (TOTP, push, or email).

1. Go to <https://mfa.mila.quebec>.
2. **Username** = cluster username (not your `@mila.quebec` email).
3. **Password** = the one-time registration token (not your Mila password).
4. **Immediately** add a TOTP token — scan the QR code into an authenticator app (privacyIDEA, Authy, Google Authenticator).
   After this first session, the portal only accepts TOTP codes; if you skip this step you need a new registration token from IT support.

Enroll two factors if you can (Push + TOTP) for redundancy.

### SSH setup with `milatools`

Install `uv` locally and then `milatools`, which wires up `~/.ssh/config`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --upgrade milatools
mila init       # prompts for your cluster username, sets up SSH keys
```

This creates entries `mila`, `mila-cpu`, `mila1`..`mila5`, `cn-????` in your SSH config.
It also uploads your public key to `~/.ssh/authorized_keys` on the cluster.

Manual equivalent (if you don't want milatools) — `~/.ssh/config`:

```sshconfig
Host mila    login.server.mila.quebec
    Hostname login.server.mila.quebec
Host cn-????
    Hostname %h.server.mila.quebec
Match host !*login.server.mila.quebec,*.server.mila.quebec
    ProxyJump mila
Match host *login.server.mila.quebec
    Port 2222
    ServerAliveInterval 120
Match host *.server.mila.quebec
    PreferredAuthentications publickey,keyboard-interactive
    AddKeysToAgent yes
    User YOUR_MILA_USERNAME
```

### Connecting

```bash
ssh mila                                   # land on a login node (you'll be prompted for OTP)
scp file.zip mila:scratch/uploaded.zip
rsync -avz mila:path/ local-path/
```

**Never run anything heavy on a login node** — it's shared with everyone.
Allowed: editing with vim/nano, `cp`/`mv`, `tmux`, submitting jobs, light scripts that mostly sleep.
Not allowed: training scripts, building wheels like `flash-attn` from source, anything CPU/RAM-heavy.

If you need a small interactive environment for building envs or editing:

```bash
ssh mila-cpu     # auto-allocates a CPU compute node, reuses the same job across connections
```

### Connecting to a running job's node

Once you have a job on `cn-XXXX`, from a login node: `ssh cn-XXXX`. From your laptop: `ssh -J mila USERNAME@cn-XXXX` (the `-J` jump is required because compute nodes are firewalled).

### Key concepts

- **`uv`** — fast Python package manager; drop-in replacement for pip/virtualenv.
- **`milatools` / `mila` CLI** — SSH config helper + `mila code` for VSCode on compute nodes.
- **MFA** — required for every SSH login. If the OTP prompt doesn't appear, MFA isn't set up.

---

## 2. SLURM essentials

Mila and DRAC both use SLURM. You submit jobs from a login node; they run on compute nodes.

### Core commands

| Command | Purpose |
|---|---|
| `sbatch job.sh` | Submit a batch job (non-interactive). |
| `salloc [opts]` | Request resources and drop into an interactive shell. |
| `srun [opts] cmd` | Run a command inside an allocation (also used inside batch scripts). |
| `squeue --me` (or `-u $USER`) | Show your jobs. |
| `scancel <JOBID>` | Cancel a job. `scancel -u $USER` cancels all yours. `scancel -t PD` cancels pending. |
| `scontrol show job <JOBID>` | Detailed info on a running job. |
| `sinfo` | Nodes and partitions. |
| `savail` | (Mila-custom) List available GPUs. |
| `sacct -u $USER --start=YYYY-MM-DD` | Job history. |

Max 1000 submitted jobs per user at a time.

### Common `#SBATCH` directives

```bash
#!/bin/bash
#SBATCH --ntasks=1                        # one task (one process)
#SBATCH --cpus-per-task=4                 # 4 CPU cores for that task
#SBATCH --gpus-per-task=l40s:1            # one L40S GPU; also: rtx8000:1, a100:1, or :1 for any
#SBATCH --mem-per-gpu=16G                 # or --mem=16G for total
#SBATCH --time=00:15:00                   # HH:MM:SS; after this the job is killed
#SBATCH --partition=unkillable            # see partition table
#SBATCH --output=slurm-%j.out             # %j = job id
#SBATCH --requeue                         # allow requeue after preemption
#SBATCH --signal=B:TERM@300               # send SIGTERM 5 min before time limit
```

Inside the script, run training via `srun`:

```bash
srun uv run python main.py
```

### Partitions (Mila cluster)

Priority order: `unkillable > main > long`. Higher-priority jobs can preempt lower ones;
preempted jobs are killed immediately and re-queued on the same partition.
`main` jobs never preempt each other, regardless of fair-share usage.

| Partition | Max resources | Max time | Notes |
|---|---|---|---|
| `unkillable` | 6 CPUs, 32 GB, 1 GPU | 2 days | Cannot be preempted. |
| `unkillable-cpu` | 2 CPUs, 16 GB | 2 days | CPU-only. |
| `short-unkillable` | 1000 GB, 4 GPUs | **3 hours** | 4-GPU nodes only (cn-g, cn-l, cn-n); **H100s live here**. |
| `main` | 8 CPUs, 48 GB, 2 GPUs | 5 days | Default priority. |
| `main-cpu` | 8 CPUs, 64 GB | 5 days | CPU-only. |
| `long` | unlimited | 7 days | Preemptible by `main`/`unkillable`. |
| `long-cpu` | unlimited | 7 days | CPU-only. |

A CPU-only request on `unkillable`/`main`/`long` is auto-translated to the `*-cpu` variant.

**H100s are only available on `short-unkillable`** (on `cn-n` nodes, 8 GPUs/node but only 4 bookable per job).

### GPU selection flags

```bash
--gpus-per-task=l40s:1      # specific model
--gpus-per-task=1           # any GPU model
--gres=gpu:a100:1           # same idea, older syntax
--gres=gpu:48gb:1           # at least 48 GB memory
--constraint="dgx&ampere"   # DGX A100 nodes
--constraint=nvlink         # NVLink-connected GPUs
```

Available feature tags: `turing`, `volta`, `ampere`, `hopper`, `lovelace`, `nvlink`, `dgx`, GPU-memory tags like `12gb`/`32gb`/`40gb`/`48gb`/`80gb`.

### Mila node profiles

| Nodes | GPU | Mem (GB) | #GPUs | Features |
|---|---|---|---|---|
| cn-a | RTX8000 | 48 | 8 | turing,48gb |
| cn-b | V100 | 32 | 8 | volta,nvlink,32gb |
| cn-c | RTX8000 | 48 | 8 | turing,48gb |
| cn-g | A100 | 80 | 4 | ampere,nvlink,80gb |
| cn-i001 | A100 | 80 | 4 | ampere,80gb |
| cn-j001 | A6000 | 48 | 8 | ampere,48gb |
| cn-k | A100 | 40 | 4 | ampere,nvlink,40gb |
| cn-l | L40S | 48 | 4 | lovelace,48gb |
| cn-n | **H100** | 80 | 8 | hopper,nvlink,80gb |
| cn-d | A100 (DGX) | 40/80 | 8 | ampere,nvlink,dgx |
| cn-e | V100 (DGX) | 32 | 8 | volta,nvlink,dgx,32gb |
| cn-f | CPU (rome) | — | — | — |
| cn-h | CPU (milan) | — | — | — |
| cn-m | CPU (sapphire) | — | — | — |

DGX A100 40 GB: `--gres=gpu:a100:N --constraint="dgx&ampere"`.
DGX A100 80 GB: `--gres=gpu:a100l:N --constraint="dgx&ampere"`.

### Quick interactive requests

```bash
salloc                                                  # CPU-only shell, default resources
salloc --gres=gpu:1 -c 2 --mem=12G                      # one GPU, 2 CPUs, 12G RAM
salloc --gres=gpu:1 -c 4 --mem=10G -t 12:00:00 --partition=unkillable
```

---

## 3. Storage

| Path | Perf | Purpose | Quota | Backup | Auto-cleanup |
|---|---|---|---|---|---|
| `/network/datasets/` | high | Curated read-only datasets | — | — | — |
| `/network/weights/` | high | Curated read-only model weights | — | — | — |
| `$HOME` (`/home/mila/<u>/<username>`) | low | Code, small libs | 100 GB / 1M files | daily | no |
| `$SCRATCH` (`/network/scratch/<u>/<username>`) | high | Temporary results, processed datasets | 5 TB / unlimited | no | 90 days |
| `$SLURM_TMPDIR` | **highest** | Per-job local disk | — | no | at job end |
| `/network/projects/<group>/` | fair | Long-term shared | 1 TB / 1M | daily | no |
| `$ARCHIVE` (`/network/archive/<u>/<username>`) | low | Long-term personal, login+CPU nodes only | 5 TB soft / 5.1 TB hard | **no** | no |

Check quotas: `disk-quota` (on login nodes). `df -h $ARCHIVE` for archive.

**The core data-staging pattern:** copy your dataset to `$SLURM_TMPDIR` at the start of every job.
Network storage is shared and slower; `$SLURM_TMPDIR` is node-local SSD.

```bash
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
# Or: tar -xf /network/datasets/foo.tar -C $SLURM_TMPDIR/data/
# Or: unzip /network/datasets/foo.zip -d $SLURM_TMPDIR/data/
```

For multi-node jobs, stage on one task per node:

```bash
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c \
    'mkdir -p $SLURM_TMPDIR/data && cp /network/datasets/... $SLURM_TMPDIR/data/'
```

### Shared datasets

- `/network/datasets/` — public curated datasets. Browse at <https://datasets.server.mila.quebec>.
- `/network/datasets/restricted/` — access-gated; submit a support ticket citing the access group.
- Datasets are mirrored to DRAC at `~/projects/rrg-bengioy-ad/data/curated/`.

### Sharing via ACLs

To share `$SCRATCH/X/Y/Z/` with `$USER2` (safer than `chmod 770`):

```bash
setfacl -Rdm user:${USER}:rwx  $SCRATCH/X/Y/Z/       # inheritable self-access
setfacl -Rdm user:${USER2}:rwx $SCRATCH/X/Y/Z/       # inheritable other-access (future files)
setfacl -Rm  user:${USER2}:rwx $SCRATCH/X/Y/Z/       # existing files
setfacl -m   user:${USER2}:x   $SCRATCH/X/Y/         # traversal (non-recursive!)
setfacl -m   user:${USER2}:x   $SCRATCH/X/
setfacl -m   user:${USER2}:x   $SCRATCH
```

Check with `getfacl <path>`.

---

## 4. Python environment management with `uv`

`uv` consolidates `pip`, `virtualenv`, `pip-tools` into a single fast tool.

### Install (local + cluster)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # run on both laptop and login node
```

If `uv` isn't found, reopen the shell (PATH update).

### Project workflow

```bash
uv init my-project --python=3.12        # creates pyproject.toml, .python-version, hello.py
cd my-project
uv add torch "numpy>=2.0"               # adds deps, updates pyproject.toml, syncs .venv
uv add --dev pytest                     # dev-only dep
uv remove torch
uv sync                                  # reproduce .venv from uv.lock
uv cache prune                           # free disk (stale cache only)
uv cache clean                           # wipe cache entirely
```

**Commit `uv.lock`** to version control — it pins exact transitive versions for reproducibility.

### Running code

```bash
uv run python train.py                   # syncs env, runs in project venv
uv run pytest
```

No need to manually activate `.venv`; `uv run` does it each time.

### Standalone scripts (PEP 723)

```bash
uv add --script experiment.py numpy matplotlib
uv run --script experiment.py
```

Declares deps in a `# /// script` block at the top of the file; `uv run` creates a temp env, runs, discards.

### CLI tools (persistent, isolated)

```bash
uv tool install --upgrade wandb
uv tool install --upgrade milatools
uv tool list
uv tool upgrade --all
```

Use `uv tool install` for commands you want globally (`wandb`, `mila`). Use `uv add` for libraries your code imports.

### In a SLURM job

```bash
#!/bin/bash
#SBATCH --gpus-per-task=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
srun uv run python train.py
```

Useful flags: `uv run --offline` (no network — required on DRAC compute nodes), `uv run --locked` (fail if lockfile out of date, enforces reproducibility).

### pip / virtualenv (fallback)

```bash
module load python/3.10
python -m venv $HOME/env
source $HOME/env/bin/activate
pip install torch torchvision
```

Modules: `module avail`, `module load python/3.12`, `module spider <name>`.

---

## 5. Experiment tracking with WandB

Mila has a team WandB organization — request access at `it-support@mila.quebec`.

### Sign in (first time)

Go to <https://wandb.ai>, enter your `@mila.quebec` address, click **Log in** — SSO will redirect to Mila's IDP.
Choose **Professional** at account creation. If you already had a personal WandB account, add `@mila.quebec` as primary email first to avoid duplicates.

### CLI auth on the cluster

```bash
uv tool install --upgrade wandb
wandb login                          # paste API key from wandb.ai/authorize; stored in ~/.netrc
```

### Init + logging pattern

```python
import os, wandb

wandb.init(
    project="my-project",
    name=os.environ.get("SLURM_JOB_ID"),
    id=os.environ.get("SLURM_JOB_ID"),              # lets us resume the same run
    group=os.environ.get("SLURM_ARRAY_JOB_ID"),     # groups array-task runs in the UI
    resume="allow",
    tags=["resnet18", "example"],
    config=vars(args)
    | {f"env/{k}": v for k, v in os.environ.items() if k.startswith("SLURM")},
)

wandb.log({"train/loss": loss, "train/accuracy": acc})
# once per epoch:
wandb.log({"val/loss": vl, "val/accuracy": va, "epoch": epoch})

wandb.finish()   # explicit is safer in SLURM jobs
```

Prefix metric names with `train/` and `val/` — the UI groups matching prefixes into the same chart.

### Offline mode (DRAC, or no-internet compute nodes)

```bash
export WANDB_MODE=offline
srun uv run --offline python main.py
# after the job:
wandb sync --sync-all
```

On DRAC, alternatively: `module load httpproxy` before `srun`, then log online.
**Caution on DRAC:** `httpproxy` + WandB artifacts is buggy — the logger hangs until the job times out. Prefer offline mode.

### Diagnosing bottlenecks

WandB auto-logs GPU/CPU/memory under the **System** tab. A sustained ~100% GPU utilization → compute-bound (good).
Low or oscillating utilization → I/O-bound: fix by increasing `DataLoader(num_workers=...)`, `pin_memory=True`, or staging data to `$SLURM_TMPDIR`.

---

## 6. Training examples

### 6.1 Single-GPU training (the baseline)

**`job.sh`:**

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:15:00
set -e
echo "Date: $(date)"; echo "Hostname: $(hostname)"

# Stage dataset to local SSD
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/

srun uv run python main.py
```

**`main.py` (skeleton):**

```python
import argparse, logging, os, random, sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.random.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)

    model = resnet18(num_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)

    data_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "data"
    train_ds, val_ds, test_ds = make_datasets(str(data_path))

    num_workers = get_num_workers()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=num_workers, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              num_workers=num_workers, shuffle=False)

    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}",
                         disable=not sys.stdout.isatty()):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        val_loss, val_acc = validation_loop(model, val_loader, device)
        print(f"Epoch {epoch}: val_loss={val_loss:.3f} val_acc={val_acc:.2%}")


@torch.no_grad()
def validation_loop(model, loader, device):
    model.eval()
    total_loss, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y).item()
        n += x.shape[0]
        correct += logits.argmax(-1).eq(y).sum().item()
    return total_loss, correct / n


def make_datasets(path, val_split=0.1, seed=42):
    train = CIFAR10(root=path, transform=transforms.ToTensor(), download=True, train=True)
    test  = CIFAR10(root=path, transform=transforms.ToTensor(), download=True, train=False)
    n_val = int(val_split * len(train))
    train, val = random_split(train, (len(train) - n_val, n_val),
                               torch.Generator().manual_seed(seed))
    return train, val, test


def get_num_workers():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else torch.multiprocessing.cpu_count()


if __name__ == "__main__":
    main()
```

### 6.2 Multi-GPU (single-node DDP) — diff from single-GPU

**`job.sh`:**

```diff
-#SBATCH --ntasks=1
-#SBATCH --ntasks-per-node=1
+#SBATCH --ntasks=2
+#SBATCH --ntasks-per-node=2
 #SBATCH --gpus-per-task=l40s:1

+export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
+export MASTER_ADDR="127.0.0.1"

-srun uv run python main.py
+srun --gres-flags=allow-task-sharing uv run python main.py
```

`--gres-flags=allow-task-sharing` is **required** when using `--gpus-per-task=1`; without it, NCCL fails with `ncclUnhandledCudaError` because cgroups isolate each task's shared memory (see slurm.schedmd.com bug 17875).

**Key code changes in `main.py`:**

```python
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.utils.data.distributed import DistributedSampler

def setup():
    rank       = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    return rank, world_size

# In main():
rank, world_size = setup()
is_master = rank == 0
device = torch.device("cuda", 0)  # SLURM sets CUDA_VISIBLE_DEVICES per-task
model = nn.parallel.DistributedDataParallel(model.to(device),
                                             device_ids=[0], output_device=0)

train_sampler = DistributedSampler(train_ds, shuffle=True)
train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                          shuffle=False, num_workers=num_workers)

for epoch in range(epochs):
    train_sampler.set_epoch(epoch)   # required so shuffle order varies
    # ...
    # Reduce metrics across ranks:
    n_correct = local_n_correct.clone()
    dist.reduce(n_correct, dst=0, op=ReduceOp.SUM)
```

In `make_datasets()`, let only the master rank download, others wait on `dist.barrier()` before reading from disk.

### 6.3 Multi-node DDP — diff from multi-GPU

```diff
-#SBATCH --ntasks=2
+#SBATCH --ntasks=4
 #SBATCH --ntasks-per-node=2
 #SBATCH --gpus-per-task=l40s:1

-mkdir -p $SLURM_TMPDIR/data && cp ... $SLURM_TMPDIR/data/
+srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c \
+   'mkdir -p $SLURM_TMPDIR/data && cp /network/datasets/... $SLURM_TMPDIR/data/'

-export MASTER_ADDR="127.0.0.1"
+export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
```

**In `setup()`:** use TCP init (the `env://` init method uses file locks and deadlocks under Mila's distributed FS config).

```python
local_rank  = int(os.environ["SLURM_LOCALID"])   # for data-download sync (per-node)
dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    timeout=timedelta(seconds=60),
    world_size=world_size, rank=rank,
)
```

Use `local_rank == 0` (one process per node) to decide who downloads data on each node.

### Alternative: `torchrun` for multi-node

```bash
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4

RDV_ADDR=$(hostname)

srun --label torchrun \
   --nproc_per_node=$SLURM_GPUS_PER_NODE \
   --nnodes=$SLURM_JOB_NUM_NODES \
   --rdzv_id=$SLURM_JOB_ID \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$RDV_ADDR \
   training_script.py
```

### 6.4 Checkpointing and preemption

Preemption on the `long` or `main-grace`/`long-grace` partitions sends signals before kill. Trap them and save state.

**`job.sh`:**

```bash
#SBATCH --requeue
#SBATCH --signal=B:TERM@300           # SIGTERM 5 min before time limit

exec srun uv run python main.py       # exec so signals reach Python directly
```

**In `main.py`:**

```python
import signal, shutil, uuid, torch
from pathlib import Path

SCRATCH    = Path(os.environ["SCRATCH"])
SLURM_JOBID = os.environ["SLURM_JOBID"]
CKPT_DIR   = SCRATCH / "myproj" / SLURM_JOBID / "checkpoints"
CKPT_FILE  = "checkpoint.pth"

def handle_preempt(signum, frame):
    logger.error(f"Received {signal.Signals(signum).name}")
    # Quick action; don't checkpoint mid-epoch here.
    # If using wandb: wandb.mark_preempting()

signal.signal(signal.SIGTERM, handle_preempt)   # before requeue
signal.signal(signal.SIGUSR1, handle_preempt)   # before time limit

# Atomic save: write to temp, then os.replace
def save_checkpoint(state, is_best):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_DIR / f"{CKPT_FILE}.tmp{uuid.uuid1()}"
    torch.save(state, tmp)
    os.replace(tmp, CKPT_DIR / CKPT_FILE)
    if is_best:
        shutil.copyfile(CKPT_DIR / CKPT_FILE, CKPT_DIR / "model_best.pth")

# At the top of training, resume if possible
if (CKPT_DIR / CKPT_FILE).exists():
    state = torch.load(CKPT_DIR / CKPT_FILE, weights_only=False)
    start_epoch = state["epoch"] + 1
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    # Also restore RNG states (random, np.random, torch, torch.cuda) for reproducibility
```

Save the full RunState at each epoch: `epoch`, `best_acc`, `model_state`, `optimizer_state`,
`random_state`, `numpy_random_state`, `torch_random_state`, `torch_cuda_random_state`.
Check `SLURM_RESTART_COUNT` to detect that SLURM requeued you — if a checkpoint isn't there despite restarts, your checkpointing code isn't running before preemption.

### 6.5 Passing args to jobs

```bash
# job.sh
srun uv run python main.py "$@"

# Submit:
sbatch job.sh --learning-rate 0.1
sbatch job.sh --learning-rate 0.5 --weight-decay 1e-3
```

### 6.6 SLURM job arrays (hyperparameter sweeps)

```bash
sbatch --array=1-100 --gres=gpu:1 --cpus-per-gpu=2 --mem-per-gpu=16G job.sh train.py
```

`$SLURM_ARRAY_JOB_ID`, `$SLURM_ARRAY_TASK_ID` are exported automatically. Use them to pick hyperparameters, seed, and (with WandB) `group=$SLURM_ARRAY_JOB_ID`.

### 6.7 Hyperparameter optimization with Orion

Orion stores trials in a pickleddb on shared storage. Use it with job arrays.

```bash
export ORION_CONFIG=$SLURM_TMPDIR/orion-config.yml
cat > $ORION_CONFIG <<EOM
experiment:
  name: orion-example
  algorithms:
    tpe: { seed: null, n_initial_points: 5 }
  max_broken: 10
  max_trials: 10
storage:
  database:
    host: $SCRATCH/orion.pkl
    type: pickleddb
EOM

srun uv run orion hunt --config $ORION_CONFIG python main.py \
    --learning-rate~'loguniform(1e-5, 1.0)'
```

In your script, report the objective at the end:

```python
from orion.client import report_objective
report_objective(1 - val_accuracy)    # Orion minimizes, so convert accuracy
```

### 6.8 Packing multiple processes on one GPU

Share a GPU between N tasks:

```bash
#SBATCH --ntasks=2 --gres=gpu:1 --cpus-per-task=4 --mem=18G
srun --label --output=out_%t.out python script "$@"
```

Or multi-prog:

```bash
srun --label --multi-prog config.conf
# config.conf:
# 0   python script firstarg
# 1   python script secondarg
```

One task per GPU on a full node (4 GPUs):

```bash
#SBATCH --nodes=1 --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=6
srun --gres=gpu:1 --ntasks=1 --mem=30G --exclusive python script args1 &
srun --gres=gpu:1 --ntasks=1 --mem=30G --exclusive python script args2 &
srun --gres=gpu:1 --ntasks=1 --mem=30G --exclusive python script args3 &
srun --gres=gpu:1 --ntasks=1 --mem=30G --exclusive python script args4 &
wait
```

`--exclusive` binds each `srun` step to distinct CPUs.

---

## 7. VSCode on the cluster

Use `mila code`, which allocates a compute node and opens VSCode connected to it.

```bash
mila code CODE/my_project --alloc --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00
```

Everything after `--alloc` is passed to `salloc`. For CPU-only editing: `mila code dir --alloc --cpus-per-task=2 --mem=8G`.

**Do not** connect plain VSCode Remote-SSH to `login.server.mila.quebec` — it runs a non-trivial server process and will overload the login nodes. Use `mila-cpu` (from `mila init`) instead.

### Troubleshooting

- **VSCode stuck on "Opening Remote..."**: enable `remote.SSH.showLoginTerminal` — the OTP prompt is hidden.
- **"Cannot reconnect" lockfile errors** when connecting to multiple nodes: set `"remote.SSH.lockfilesInTmp": true` in `settings.json` — `~/.vscode-server` is shared across compute nodes.
- **Debugger timeouts**: `export DEBUGPY_PROCESS_SPAWN_TIMEOUT=500` in `~/.bashrc`.

### Selecting the Python interpreter

`Cmd/Ctrl+Shift+P` → "Python: Select interpreter" → paste the path printed by `which python` (from an activated env).

---

## 8. JupyterHub

Available at <https://jupyterhub.server.mila.quebec> (Google OAuth with `@mila.quebec`).
Launches a JupyterLab session as a SLURM job automatically.

**Close sessions via `Hub → Control Panel → Stop my server`** — closing the window leaves the SLURM job running.

JupyterLab cannot navigate above `$HOME`, so `$SLURM_TMPDIR` and `/network/datasets` are invisible. Create symlinks:

```bash
ln -s /network/datasets $HOME
ln -sf $SLURM_TMPDIR $HOME         # needs to be redone each session
```

---

## 9. Containers

### Podman (preferred on Mila)

Docker doesn't run on the cluster (security). Use Podman inside SLURM jobs — compatible with Docker CLI.

```bash
podman run --mount type=bind,source=$SCRATCH/exp,destination=/data/exp bash touch /data/exp/file

# With GPUs:
podman run --device nvidia.com/gpu=all nvidia/cuda:11.6.1-base-ubuntu20.04 nvidia-smi
# Or specific GPU indices:
podman run --device nvidia.com/gpu=0 ...
```

Storage: Podman images/containers live in a job-specific location that's wiped at job end.
Bind-mount `$SCRATCH` or `$HOME` for persistent data.

Expect benign warnings like `WARN[0000] "/" is not a shared mount` and `cannot find UID/GID`.

### Singularity (on DRAC)

Docker is unavailable; Singularity is standard on Alliance clusters. Build images elsewhere
(build is CPU-heavy — don't do it on login nodes). Run pre-built images from a `.sif` file.

```bash
# Pull a public image (on a compute node with internet, not login)
singularity pull docker://pytorch/pytorch:2.0-cuda11.7-cudnn8-runtime

# Use in a job
module load singularity      # DRAC only
rsync -avz $SCRATCH/my.sif $SLURM_TMPDIR
singularity exec --nv \
  -H $HOME:/home \
  -B $SLURM_TMPDIR:/dataset/ \
  -B $SCRATCH:/final_log/ \
  $SLURM_TMPDIR/my.sif python main.py
```

`--nv` enables GPU passthrough. `-H` sets the container home; `-B src:dst` binds host paths.

---

## 10. DRAC (Digital Research Alliance of Canada) clusters

For larger/longer jobs than Mila allows. Main clusters and GPU types:

| Cluster | Successor to | GPU | Notes |
|---|---|---|---|
| Fir | Cedar | H100-80G | Retains Cedar's FS. |
| Nibi | Graham | H100-80G | Retains Graham's FS. |
| Rorqual | Beluga | H100-80G | **No internet on compute nodes.** |
| Trillium | Niagara | H100 | Mostly CPU, per-node (not per-CPU) allocation. Not generally recommended. |
| Narval | — | A100-40G | Oldest. Good for sub-H100 workloads. |
| TamIA | — | H100/H200 | PAICE AI cluster. Uses `--account=aip-${PI}` (not `rrg-`). Per-node allocation. |

### Account setup

1. Register at <https://ccdb.alliancecan.ca> (password ≥ 8 chars, mixed case, digits, special).
2. Apply for a `role` at <https://ccdb.alliancecan.ca/me/add_role> — you need your sponsor's CCRI (ask supervisor). Apply once per allocation (`rrg-bengioy-ad`, `def-<supervisor>`).
3. Wait for the sponsor to accept. Accounts must be renewed annually.

### Running jobs on DRAC

You must specify an `--account`:

```bash
sbatch --time=1:00:00 --account=rrg-bengioy-ad --gres=gpu:1 job.sh
salloc --time=1:00:00 --account=rrg-bengioy-ad --gres=gpu:1
```

### DRAC storage

| Path | Purpose |
|---|---|
| `$HOME` (`/home/<user>`) | Code, libraries (read once). |
| `$HOME/projects/rrg-bengioy-ad` | Compressed raw datasets. |
| `$SCRATCH` (`/scratch/<user>`) | Processed datasets, results. Purged at ~3 months. |
| `$SLURM_TMPDIR` | Per-job local disk. |

### DRAC Python setup inside a job

```bash
module load StdEnv/2023 python/3.12
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch torchvision     # use precompiled wheels

# Copy dataset
unzip $SCRATCH/dataset.zip -d $SLURM_TMPDIR

python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR
cp $SLURM_TMPDIR/results $SCRATCH
```

Available precompiled wheels: `avail_wheels <name>`.

### DRAC internet access

Compute nodes on Narval/Rorqual/TamIA have no internet. Options for wandb/comet:

```bash
module load httpproxy
```

or run offline and `wandb sync --sync-all` from the login node afterwards.

### RGUs

DRAC measures allocation in Reference GPU Units, combining FP32/FP16 perf and memory. Rough guide: A100-40G = 4 RGU, H100-80G = 12.15 RGU.

---

## 11. AI assistants (Claude Code)

### Install

```bash
# CLI (macOS, Linux, WSL2)
curl -fsSL https://claude.ai/install.sh | bash
claude --version
```

Also available as a VS Code extension. See <https://claude.com/product/claude-code>.

### Auth

Run `claude` — a browser window opens for login. Requires an Anthropic account (Pro/Team/Enterprise subscription or API-usage billing).

### Skills

Skills are plugin workflows triggered as slash commands.

```
/plugin marketplace add mila-iqia/skills
/plugin install mila-tools@mila-skills
/reload-plugins
```

Available Mila skills (via `mila-iqia/skills`):

| Skill | Purpose |
|---|---|
| `/mila-account-setup` | Onboard: account, cluster access, MFA. |
| `/mila-local-setup` | WSL2, uv, milatools on your laptop. |
| `/mila-connect-cluster` | SSH + OTP + troubleshoot connectivity. |
| `/mila-run-jobs` | Interactive sessions + sbatch submission. |

Manage plugins with `/plugin` (Discover / Installed / Marketplaces / Errors tabs).

---

## 12. Cluster concepts (brief theory)

- **Login vs. compute nodes.** Login = shared entry point, no heavy compute. Compute = where jobs run, often with GPUs.
- **Storage nodes.** Dedicated machines serving `/network/*`; invisible to you except via paths.
- **Workload manager (SLURM).** You submit jobs to a queue; SLURM matches them to resources using policy (priority, fair-share, time since submission, resources asked). Partitions group nodes by purpose.
- **Preemption.** Soft limits → your job gets `SIGTERM` (with possible grace period) when a higher-priority job needs the node. Hard limits → immediate kill. Mila's `-grace` partitions give you 120 s before kill (but don't auto-requeue; call `sbatch` inside a trap handler if you want that).
- **Data vs. model parallelism.** Data parallelism = N workers, each processes a shard — easiest, use first (DDP). Model parallelism = one model split across devices — needed only when a model doesn't fit on a GPU. Communication patterns: infrequent (data parallel) → tasks can be anywhere; frequent (model parallel) → colocate via `--nodes`, NVLink/InfiniBand matters.
- **Filesystem performance.** Different mounts are tuned for different I/O patterns. A bad choice (e.g. training from `$HOME`) can be 100x slower. Use `$SLURM_TMPDIR` for hot paths.

---

## 13. FAQ (condensed)

### Connection / SSH

- **`connection refused`**: the login node banned your IP after too many failed connects. Unbanned automatically after 1 hour.
- **`Permission denied (publickey)` on a compute node**: add your pubkey to `~/.ssh/authorized_keys` on the cluster or run an `ssh-agent` that forwards it. `ssh-keygen` on a login node and append to `authorized_keys`.
- Mila compute-node SSH fingerprints (to avoid MITM warnings):
  ```
  SHA256:hGH64v72h/c0SfngAWB8WSyMj8WSAf5um3lqVsa7Cfk (ECDSA)
  SHA256:4Es56W5ANNMQza2sW2O056ifkl8QBvjjNjfMqpB7/1U (RSA)
  SHA256:gUQJw6l1lKjM1cCyennetPoQ6ST0jMhQAs/57LhfakA (ED25519)
  ```

### SLURM

- **Job stuck with `ReqNodeNotAvail`**: no matching node free. Not an error — wait, or `scancel` and resubmit with a different GPU type (`--gres=gpu:rtx8000:1`).
- **"Invalid account or account/partition combination"**: your account isn't set up; file a ticket.
- **`srun: error: --mem and --mem-per-cpu are mutually exclusive`**: benign; `salloc` added a default. Ignore.
- **`slurmstepd: Detected 1 oom-kill event(s)`**: your job exceeded `--mem`. Raise it or fix the leak.
- **`fork: retry: Resource temporarily unavailable`**: hit the 2000 task/PID limit. Some subprocess is spawning too many.
- **Cancel jobs**: `scancel <JOBID>` / `scancel -u $USER` (all) / `scancel -t PD` (only pending).

### PyTorch

- **Don't leave `torch.autograd.set_detect_anomaly(True)` on.** It stats every source file on every tensor creation, generating huge IOPS on the shared filesystem. Only enable interactively when debugging, then turn it off. You will be contacted if you abuse this.
- **Conda "Your installed CUDA driver is: not available"** on login/CPU nodes: set `CONDA_OVERRIDE_CUDA=11.8` (or your toolkit version) before `conda create`.

### Data

- **Where do I put data during a job?** Copy from `/network/datasets/` or `$SCRATCH` to `$SLURM_TMPDIR` at job start. Save outputs you want to keep back to `$SCRATCH` before the job ends — `$SLURM_TMPDIR` is deleted on job end.

### Shell

- Default shell is `/bin/bash`. To change, file a ticket. `zsh` is supported; other shells are not supported.

---

## 14. Help and support

Order of escalation:

1. **This doc / official docs** (<https://docs.mila.quebec>)
2. **On-site chatbot** (the "Ask AI" button in the docs site)
3. **Slack** `#mila-cluster` (Mila's Slack) — fastest for community help
4. **Helpdesk** (IT support portal: <https://it-support.mila.quebec>) — tickets for account/access/SSH issues
5. **IDT Office Hours** — in-person (Lab A) or online (mila-core calendar) for debugging + ML workflow help

Team routing:

| Issue | Contact |
|---|---|
| Account creation, SSH, VPN, hardware | IT Support (portal or `#mila-cluster`) |
| Job debugging, experiment optimization | IDT (`#mila-cluster`, office hours) |
| General admin, community | MyMila portal |

---

## 15. Abbreviations / Glossary

- **cluster** — a group of computers that run jobs together.
- **login node** — the entry point you SSH into; don't run heavy work here.
- **compute node** — where jobs actually run; GPUs live here.
- **GPU** — graphics processing unit; what accelerates DL training.
- **SLURM** — the job scheduler.
- **SLURM job** — a submitted unit of work, with resources and a time limit.
- **SLURM job step** — an individual command/program inside a job (via `srun`).
- **batch SLURM job** — a job submitted via `sbatch` that runs without you attached.
- **MFA** — multi-factor authentication: more than one credential required.
- **SSH** — Secure SHell; the connection protocol.
- **OTP** — one-time password.
- **TOTP** — time-based OTP (6-digit code from an authenticator app).
- **WSL** — Windows Subsystem for Linux (for Windows users).
- **DRAC / Alliance** — Digital Research Alliance of Canada (national compute).
- **DDP** — Distributed Data Parallel (PyTorch's multi-GPU training).
- **RGU** — Reference GPU Unit (DRAC's weighted GPU allocation currency).
- **HPO** — hyperparameter optimization.

---

*Derived from <https://docs.mila.quebec>. See the original for full context, extended examples, and the authoritative cheatsheet PDF.*
