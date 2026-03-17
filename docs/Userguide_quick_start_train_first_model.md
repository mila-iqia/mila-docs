# Train Your First Model

This guide walks you through training a small model (ResNet18) on CIFAR-10 using
a single GPU on the Mila cluster. You will use Mila's CIFAR-10 dataset, stage it
into fast local storage, and run a Slurm batch job.

## Prerequisites

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Quick Start__](Userguide_quick_start.md)
    { .card }

    ---

    Quick start

-   [:material-run-fast:{ .lg .middle } __Run your first job__](Userguide_quick_start_my_first_job.md)
    { .card }

    ---

    Run your first job on the cluster with PyTorch

</div>

## What you will do

* Train a ResNet18 on CIFAR-10 using a single GPU.
* Use Mila's CIFAR-10 dataset in `/network/datasets/cifar10/`.
* Stage data into `$SLURM_TMPDIR` for fast I/O during training.
* Submit and monitor a batch job with `sbatch`.

---

## Open VSCode on a compute node

### Create the project directory on the cluster

From your **local machine**, create the project directory on the cluster so that
`mila code` can open it (the path is on the cluster):

```bash
ssh mila 'mkdir -p CODE/train_first_model'
```

### Start VSCode on a CPU node

For this step, we're only preparing code and job scripts—not actually running
training—so we'll use a CPU node for a faster to allocate and less
resource-intensive editor session.

```bash
mila code CODE/train_first_model --alloc --cpus-per-task=2 --mem=16G --time=01:00:00
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
[17:35:21] Checking disk quota on $HOME...                                                                                                     disk_quota.py:31
[17:35:27] Disk usage: 85.34 / 100.00 GiB and 794022 / 1048576 files                                                                           disk_quota.py:211
[17:35:29] (mila) $ cd $SCRATCH && salloc --cpus-per-task=2 --mem=16G --time=01:00:00 --job-name=mila-code                                     compute_node.py:293
salloc: --------------------------------------------------------------------------------------------------
salloc: # Using default long partition
salloc: --------------------------------------------------------------------------------------------------
salloc: Granted job allocation 8888888
[17:35:30] Waiting for job 8888888 to start.                                                                                                   compute_node.py:315
[17:35:31] (localhost) $ code --new-window --wait --remote ssh-remote+cn-a003.server.mila.quebec /home/mila/u/username/CODE/train_first_model  local_v2.py:55
```
</div>

### Create the project files

=== "job.sh"

    The job script does three things:
    
    1. **`#SBATCH` directives** — Request 1 GPU, 4 CPUs, 16G memory, 15 minutes.
    2. **Data staging** — Copy CIFAR-10 from `/network/datasets/` into `$SLURM_TMPDIR/data`. Compute nodes read from `$SLURM_TMPDIR` much faster than from network storage.
    3. **Run the training script** — `srun uv run python main.py` runs your script inside the allocation.

    ```bash title="job.sh"
    --8<-- "docs/examples/distributed/single_gpu/job.sh"
    ```

=== "pyproject.toml"

    ```toml title="pyproject.toml"
    --8<-- "docs/examples/distributed/single_gpu/pyproject.toml"
    ```

=== "main.py"

    The training script uses PyTorch to load CIFAR-10 from `$SLURM_TMPDIR/data`,
    train ResNet18, and log validation loss and accuracy per epoch. Create a
    file `main.py` with the following content:

    ```py title="main.py"
    --8<-- "docs/examples/distributed/single_gpu/main.py"
    ```

## Submit the job

Using the VSCode terminal, submit the job:

```bash
sbatch job.sh
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
sbatch: --------------------------------------------------------------------------------------------------
sbatch: # Using default long partition
sbatch: --------------------------------------------------------------------------------------------------
Submitted batch job 8888888
```
</div>

You will see something like `Submitted batch job 8888888`. Note the job ID.

## Monitor the job

* **Queue status:** `squeue --me`
  ```bash
  squeue --me
  ```
  <div class="result" style="border:None; padding:0" markdown>
  ``` linenums="0"
     JOBID     USER    PARTITION           NAME  ST START_TIME             TIME NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) COMMENT
   8888888 username         long         job.sh   R 2026-03-16T19:48       1:08     1    4        N/A     16G cn-l016 (None) (null)
  ```
  </div>
  
* **Watch the output file:** Once the job starts, a file `slurm-<JOBID>.out`
  will be created in which you'll find the log of the job being executed. Open
  it to watch the model being trained:
  ``` title="slurm-JOBID.out"
  Date:     Tue Mar 17 02:55:56 PM EDT 2026
  Hostname: cn-l023.server.mila.quebec
  Using CPython 3.12.11
  Creating virtual environment at: .venv
  Installed 36 packages in 1m 56s
  [03/17/26 14:58:37] INFO     Epoch 0: Val loss: 59.757 accuracy: 46.80%                                      main.py:152
  [03/17/26 14:58:40] INFO     Epoch 1: Val loss: 42.472 accuracy: 62.78%                                      main.py:152
  [03/17/26 14:58:42] INFO     Epoch 2: Val loss: 41.969 accuracy: 63.88%                                      main.py:152
  [03/17/26 14:58:45] INFO     Epoch 3: Val loss: 47.068 accuracy: 60.84%                                      main.py:152
  [03/17/26 14:58:48] INFO     Epoch 4: Val loss: 38.446 accuracy: 67.44%                                      main.py:152
  [03/17/26 14:58:50] INFO     Epoch 5: Val loss: 39.577 accuracy: 67.66%                                      main.py:152
  [03/17/26 14:58:53] INFO     Epoch 6: Val loss: 40.347 accuracy: 68.30%                                      main.py:152
  [03/17/26 14:58:56] INFO     Epoch 7: Val loss: 46.434 accuracy: 67.30%                                      main.py:152
  [03/17/26 14:58:58] INFO     Epoch 8: Val loss: 44.021 accuracy: 69.44%                                      main.py:152
  [03/17/26 14:59:01] INFO     Epoch 9: Val loss: 47.403 accuracy: 69.00%                                      main.py:152
  Done!
  ```

## Key concepts

* **Data staging to `$SLURM_TMPDIR`** — Network storage is shared and slower.
  Copying the dataset into `$SLURM_TMPDIR` at the start of the job gives the
  compute node fast local access for the rest of the run.
* **srun** — Runs a command inside the allocated resources. In our script, `srun
  uv run python main.py` runs the training on the GPU node.
