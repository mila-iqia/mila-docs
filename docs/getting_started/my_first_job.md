---
title: Run Your First Job
description: >-
  Create a minimal PyTorch project, open VSCode on a GPU compute node
  using mila code, and verify CUDA availability.
skills:
  - skill-mila-run-jobs
---

# Run Your First Job

This guide covers running a first job on the Mila cluster. Create a minimal
PyTorch project that checks CUDA and GPU availability, and develop on the
cluster using [VSCode](VSCode.md) on a compute node via the [`mila
code`](https://github.com/mila-iqia/milatools) command from
[milatools](https://github.com/mila-iqia/milatools).

## Prerequisites

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Get Started with the Cluster__](index.md)
    { .card }

    ---

    Get your Mila account, enable cluster access and MFA, then install `uv` and
    `milatools` to connect via SSH.

&nbsp;

</div>

!!! success "VSCode or compatible editor"

    The `mila code` command opens [VSCode](https://code.visualstudio.com/) (or a
    compatible editor such as Cursor) on a compute node. [Install
    VSCode](https://code.visualstudio.com/Download) on your local machine before
    starting.

## What this guide covers

* Open VSCode on a compute node with one GPU using `mila code` (from your local
  machine).
* Create a minimal PyTorch project with `pyproject.toml` and `main.py`.
* Run the script with `uv run python main.py` in VSCode.

---

## Open VSCode on a compute node

### Create the project directory on the cluster

From your **local machine**, create the project directory on the cluster so that
`mila code` can open it (the path is on the cluster):

```bash
ssh mila 'mkdir -p CODE/my_first_job'
```

### Start VSCode on a GPU node

Run `mila code` with allocation options to request one GPU. This allocates a
compute node and opens VSCode in a project path; everything after `--alloc` is
passed to Slurm:

```bash
mila code CODE/my_first_job --alloc --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
[17:35:21] Checking disk quota on $HOME...                                                                                                disk_quota.py:31
[17:35:27] Disk usage: 85.34 / 100.00 GiB and 794022 / 1048576 files                                                                      disk_quota.py:211
[17:35:29] (mila) $ cd $SCRATCH && salloc --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --job-name=mila-code                   compute_node.py:293
salloc: --------------------------------------------------------------------------------------------------
salloc: # Using default long partition
salloc: --------------------------------------------------------------------------------------------------
salloc: Granted job allocation 8888888
[17:35:30] Waiting for job 8888888 to start.                                                                                              compute_node.py:315
[17:35:31] (localhost) $ code --new-window --wait --remote ssh-remote+cn-a003.server.mila.quebec /home/mila/u/username/CODE/my_first_job  local_v2.py:55
```
</div>

Wait until the allocation is granted and VSCode opens, connected to a shell on
the compute node.

## Run a script in VSCode

### Create the project files

In VSCode, create the following two files in the project folder (e.g. in the
explorer or via **File → New File**). The files live on the compute node.

=== "pyproject.toml"

    ```toml title="pyproject.toml"
    --8<-- "docs/examples/frameworks/pytorch_setup/pyproject.toml"
    ```

=== "main.py"

    ```py title="main.py"
    --8<-- "docs/examples/frameworks/pytorch_setup/main.py"
    ```

### Run the script in the VSCode terminal

Open the integrated terminal in VSCode (**Terminal → New Terminal**). You are on
the compute node. From the project directory, run:

```bash
uv run python main.py
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
Using CPython 3.12.11
Creating virtual environment at: .venv
Installed 28 packages in 12.08s
PyTorch built with CUDA:         True
PyTorch detects CUDA available:  True
PyTorch-detected #GPUs:          1
    GPU 0:      Quadro RTX 8000
```
</div>

The output should confirm that PyTorch is built with CUDA and detects the GPU.
When done, close VSCode and press **Ctrl+C** in the terminal to end the `mila
code` session and relinquish the allocation.

## Key concepts

`mila code`
:   Allocates a compute node and opens VSCode on it. Use for interactive
    development and running scripts with a full editor and terminal.

---

## Next step

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Train Your First Model__](train_first_model.md)
    { .card }

    ---

    Train a ResNet18 on CIFAR-10 on a single GPU using `sbatch`.

&nbsp;

</div>
