---
title: Manage Python Dependencies with uv
description: >-
  Set up uv, declare dependencies, and run reproducible Slurm jobs.
---

# Manage Python Dependencies with `uv`

`uv` is a fast Python package and project manager that consolidates `pip`,
`virtualenv`, and `pip-tools` into a single tool. This guide covers installing
`uv` locally and on the cluster, managing project dependencies, running scripts
reproducibly, and submitting Slurm batch jobs.

## Before you begin

<div class="grid cards" markdown>

-   [:material-server:{ .lg .middle } __Get Started with the Cluster__](../getting_started/index.md)
    { .card }

    ---
    Obtain a Mila account, set up MFA, install `uv` and `milatools`, and connect to
    the cluster for the first time.

&nbsp;

</div>

## What this guide covers

* Install `uv` on a local machine and on the Mila cluster
* Create a project and declare dependencies in `pyproject.toml`
* Add, update, and remove packages
* Understand how `uv` resolves dependencies safely
* Reproduce the environment on a new node using `uv.lock`
* Run standalone scripts with inline dependencies
* Submit a Slurm job that runs your script with `uv run python`

---

## Install `uv`

### On a local machine

Run the following command in a terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
downloading uv 0.10.10 x86_64-unknown-linux-gnu
no checksums to verify
installing to /home/username/.local/bin
  uv
  uvx
everything's installed!
```
</div>

Verify the installation:

```bash
uv --version
```

<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
uv 0.10.10
```
</div>

!!! note "If `uv` is not found"
    If `uv --version` returns `command not found`, close and reopen your
    terminal to apply the updated `PATH`.

### On the cluster

Connect to a login node:

```bash
ssh mila
```

Then run the install command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

!!! note "If `uv` is not found"
    If `uv --version` returns `command not found`, exit the SSH session and
    reconnect to apply the updated `PATH`.

## Create a project

Initialize a new project with a pinned Python version:

```bash
uv init my-project --python=3.12
cd my-project
```

<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
Initialized project `my-project` at `my-project`
```
</div>

The command creates the following files:

```
my-project/
├── pyproject.toml    # project metadata and dependencies
├── .python-version   # pinned Python version
├── hello.py          # sample entry point
└── README.md
```

The virtual environment (`.venv/`) and lockfile (`uv.lock`) are created
the first time `uv add` or `uv run` is called.

??? tip "Initialize a project in an existing directory"
    Run `uv init` without a directory name from inside an existing directory to
    initialize a project in place.

## Manage dependencies

### Add packages

`uv add` installs a package, updates `pyproject.toml`, and synchronizes
the virtual environment in a single step:

```bash
uv add torch
```

<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
Resolved 7 packages in 1.23s
Prepared 5 packages in 45.67s
Installed 5 packages in 1.23s
 + torch==2.5.1
```
</div>

Specify version constraints using PEP 440 syntax:

```bash
uv add "numpy>=2.0"
```

### How `uv` resolves dependencies

`pip` installs packages one command at a time and only checks constraints for
the packages in that command — it can silently break packages that are already
installed:

> "when you run a `pip` install command, `pip` only considers the packages you
  are
> installing in that command, and may break already-installed packages."
> — [`pip`
  documentation](https://pip.pypa.io/en/stable/user_guide/#watch-out-for)

`uv add` resolves the **entire dependency graph** before writing anything to
disk. If adding a new package would conflict with an existing one, `uv` reports
the conflict and aborts rather than leaving the environment in a broken state.

### Add development dependencies

Development dependencies — testing tools, linters, formatters — are only needed
during local development, not when running your research code. Add them with
`--dev`:

```bash
uv add --dev pytest
```

### Remove packages

```bash
uv remove torch
```

## Reproduce the environment

### Why pin exact versions

Without a lockfile, two installs of the same `pyproject.toml` can produce
different package versions — a transitive dependency may have released a new
patch between installs. This can silently change model behaviour or break code
across nodes and collaborators.

`uv.lock` records the exact resolved version of every dependency, direct and
transitive. Committing this file to version control guarantees that `uv sync`
installs a bit-for-bit identical environment on any machine.

### Sync the environment

```bash
uv sync
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
Resolved 7 packages in 0.08s
Audited 7 packages in 0.01s
```
</div>

!!! tip "Free disk space by clearing the cache"
    If your home directory is running low on space, first remove stale entries
    with:

    ```bash
    uv cache prune
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    Pruning cache at: /home/mila/u/username/.cache/uv
    Removed 21109 files (3.3GiB)
    ```
    </div>

    This removes outdated entries while keeping packages still used by your
    projects. If you need to reclaim more space, remove the entire cache with:

    ```bash
    uv cache clean
    ```
    <div class="result" style="border:None; padding:0" markdown>
    ``` linenums="0"
    Clearing cache at: /home/mila/u/username/.cache/uv
    Cleaning [========>           ] 45%
    Removed 20518 files (7.1GiB)
    ```
    </div>

    `uv` will re-download packages automatically the next time they are needed.

## Run code

### Within a project

`uv run` executes a command inside the project environment. Before running, `uv`
checks whether the lockfile matches `pyproject.toml` and updates the virtual
environment if needed. No manual activation of the virtual environment is
required.

Run a tool registered in the environment:

```bash
uv run pytest
```

Run a Python script with the project environment loaded:

```bash
uv run python train.py
```

### Standalone scripts with inline dependencies

For one-off scripts or utilities that do not belong to a project, [PEP
723](https://peps.python.org/pep-0723/) metadata lets you declare dependencies
directly inside the script file.

Use `uv add --script` to add dependencies to the script. `uv` will write the `#
/// script` metadata block automatically:

```bash
uv add --script experiment.py numpy matplotlib
```

The resulting script looks like:

```python title="experiment.py"
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)
plt.hist(data)
plt.savefig("hist.png")
```

Run the script with:

```bash
uv run --script experiment.py
```

`uv` reads the metadata block, creates a temporary isolated environment with
those exact dependencies, runs the script, and discards the environment. The
project's `pyproject.toml` is not modified.

## Use `uv` in a Slurm job

Declare the training script's dependencies in `pyproject.toml` using `uv add`:

```bash
uv add "torch>=2.5"
```

```python title="train.py"
import torch
# ... training code
```

```bash title="job.sh"
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00

srun uv run python train.py
```

Submit the job from the project root directory:

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

`uv run` syncs the environment against `uv.lock` before executing `train.py`, so
the job always uses the pinned dependencies regardless of which compute node
runs it.

---

## Key concepts

`pyproject.toml`
:   The project configuration file. Lists runtime and development dependencies,
    the required Python version, and project metadata. Updated automatically by
    `uv add` and `uv remove`.

`uv.lock`
:   A lockfile generated by `uv` that records the exact resolved version of every
    dependency. Commit this file to version control to guarantee reproducible
    environments across nodes and collaborators.

`uv run`
:   Executes a command inside the project's virtual environment. Use `uv run
    python <file>` to run a project script. Pass `--script <file>` only for
    standalone scripts with PEP 723 inline metadata. Syncs the environment
    against `uv.lock` before running, making manual activation unnecessary.

**virtual environment** (`.venv/`)
:   An isolated directory containing the Python interpreter and installed packages
    for the project. Created and managed automatically by `uv`.

**dependency group**
:   A named set of dependencies in `pyproject.toml`. The default development group
    (`--dev`) holds tools like `pytest` that are only needed during local
    development, not when running your research code on the cluster.

**PEP 723 script metadata**
:   A `# /// script` block at the top of a `.py` file that declares the script's
    Python version and dependencies. Used for one-off scripts that do not belong
    to a project. `uv add --script` writes this block automatically; `uv run
    --script` reads it and creates a temporary isolated environment, leaving
    `pyproject.toml` unchanged.
