---
name: mila-run-jobs
description: >-
  Use this skill when the user asks about running jobs or code on the Mila
  cluster. Trigger phrases include: "How do I run a job", "How do I run my
  first job", "How do I train a model", "mila code", "sbatch", "salloc",
  "VSCode on the cluster", "How do I use a GPU", "How do I request a GPU",
  "check available GPUs", "nvidia-smi", "How do I submit a batch job",
  "How do I run Python on the cluster", "How do I open VSCode on a compute
  node", "SLURM job", "squeue", "job queue", "slurm output file",
  "How do I stage data", "SLURM_TMPDIR", "How do I run PyTorch on the
  cluster", "interactive job".
version: 1.0.0
argument-hint: <interactive|batch>
---

# Running Jobs on the Mila Cluster

This skill guides users through running jobs on the Mila cluster: either
interactive development sessions using `mila code` (VSCode on a compute
node), or batch jobs submitted with `sbatch`.

## Base policies

At the start of each response, use the Read tool to load
`.claude/skills/mila-base/SKILL.md` and apply all policies defined there
before proceeding with the workflow below.

## Reference documentation

Interactive development:
**https://docs.mila.quebec/Userguide_quick_start_my_first_job/**

Batch training:
**https://docs.mila.quebec/Userguide_quick_start_train_first_model/**


## Workflow

### Step 1: Identify what the user wants

Determine which mode the user needs:

- **Interactive / exploratory** — they want to write and run code with an
  editor, debug interactively, or check GPU availability. → Use `mila code`.
- **Batch / training** — they want to submit a job that runs unattended,
  train a model overnight, or use `sbatch`. → Use `sbatch`.

If unclear, ask: "Do you want to work interactively with VSCode on a compute
node, or submit a batch job that runs on its own?"

### Step 2: Fetch the documentation

- For interactive: use the WebFetch tool to fetch **https://docs.mila.quebec/Userguide_quick_start_my_first_job/**
- For batch: use the WebFetch tool to fetch **https://docs.mila.quebec/Userguide_quick_start_train_first_model/**
- If the user asks about both, use the WebFetch tool to fetch both pages.

### Step 3a: Guide through interactive development (`mila code`)

1. From the **local machine**, create the project directory on the cluster:
   ```bash
   ssh mila 'mkdir -p CODE/my_first_job'
   ```
2. Start VSCode on a GPU compute node:
   ```bash
   mila code CODE/my_first_job --alloc --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00
   ```
   Everything after `--alloc` is passed to Slurm. Adjust resources as needed.
3. Wait for the allocation to be granted and VSCode to open, connected to
   the compute node.
4. In VSCode, create `pyproject.toml` and `main.py` in the project folder.
5. Open the VSCode integrated terminal (**Terminal → New Terminal**) — this
   terminal runs on the **compute node** — and run:
   ```bash
   uv run python main.py
   ```
6. When done, close VSCode and press **Ctrl+C** in the terminal where
   `mila code` is running to end the session and release the allocation.

Key points:
- `mila code` requires `milatools` and a working SSH connection (see
  **mila-connect-cluster**).
- VSCode must be installed locally; Cursor is also supported.
- Adjust `--gres=gpu:1`, `--mem`, and `--time` for the actual workload.

### Step 3b: Guide through batch job submission (`sbatch`)

1. From the **local machine**, create the project directory on the cluster:
   ```bash
   ssh mila 'mkdir -p CODE/train_first_model'
   ```
2. Open a CPU node for editing (faster to allocate than a GPU node):
   ```bash
   mila code CODE/train_first_model --alloc --cpus-per-task=2 --mem=16G --time=01:00:00
   ```
3. In VSCode, create three files: `job.sh`, `pyproject.toml`, and `main.py`.

   **`job.sh`** does three things:
   - `#SBATCH` directives — request resources (GPU, CPUs, memory, time).
   - Data staging — copy the dataset from `/network/datasets/` into
     `$SLURM_TMPDIR/data`. Compute nodes read from `$SLURM_TMPDIR` much
     faster than from network storage.
   - Run the training script — `srun uv run python main.py`.

4. Submit the job from the VSCode terminal:
   ```bash
   sbatch job.sh
   ```
5. Monitor the job:
   - **Queue status:** `squeue --me`
   - **Output:** once running, watch `slurm-<JOBID>.out` for logs.

Key points:
- `$SLURM_TMPDIR` is fast local storage on the compute node, available
  only during the job. Always stage datasets there before training.
- `srun` inside `job.sh` runs the command within the allocated resources.
- The Mila CIFAR-10 dataset is at `/network/datasets/cifar10/`.

### Step 4: Answer follow-up questions

Common questions:

- "How do I check if my job is still running?" — `squeue --me`
- "How do I cancel a job?" — `scancel <JOBID>`
- "How do I request multiple GPUs?" — `--gres=gpu:2` (or more) in the
  `mila code` command or `#SBATCH` directive.
- "Can I use Cursor instead of VSCode?" — Yes. `mila code` supports
  Cursor and other compatible editors.
- "What is `$SLURM_TMPDIR`?" — Fast local storage on the compute node,
  unique to each job. Use it to stage datasets for fast I/O during training.
- "How do I see job output?" — Open `slurm-<JOBID>.out` in the project
  directory once the job starts.

### Step 5: Point to further resources

For more complex job patterns (multi-GPU, distributed training, environment
variables, partitions):
- **https://docs.mila.quebec/Userguide_running_code/**
- **https://docs.mila.quebec/Userguide_multigpu/**
