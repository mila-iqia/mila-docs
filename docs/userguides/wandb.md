---
title: Track Experiments with Weights & Biases (WandB)
description: >-
  Set up WandB and follow best practices for logging experiments, organizing
  runs, and diagnosing bottlenecks on the cluster.
biel_boost: 0.5
---

# Track Experiments with Weights & Biases (WandB)

[Weights & Biases](https://wandb.ai) is an experiment tracking platform for
logging metrics, hyperparameters, and artifacts from training runs. Mila members
supervised by core professors can access the shared Mila organization on
wandb.ai for team-level project visibility and collaboration.

## Before you begin

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Train Your First Model__](../getting_started/train_first_model.md)
    { .card }

    ---
    Train your first ResNet18 model on CIFAR-10 on a single GPU using `sbatch`.

-   [:material-language-python:{ .lg .middle } __Manage Python Dependencies with `uv`__](../userguides/python_uv.md)
    { .card }

    ---
    Install uv, manage project dependencies, run reproducible Slurm jobs, and run
    standalone scripts.

</div>

!!! success "Request access to the Mila WandB organization"
    Students supervised by core professors are eligible for the Mila
    organization on wandb.ai. Write to
    [it-support@mila.quebec](mailto:it-support@mila.quebec) to request access.

## What this guide covers

* Sign in with single sign-on (SSO) using your `@mila.quebec` address.
* Authenticate the WandB CLI on the cluster.
* Initialize a run and log metrics in a training script.
* Name runs, attach tags, and group related runs.
* Configure Slurm job scripts for reliable WandB logging and run resumption.
* Identify whether a training job is I/O-bound or compute-bound using
  WandB system metrics and step timing.

---

## Sign in for the first time

WandB uses Mila's SSO provider. Signing in with your `@mila.quebec` address the
first time links the account to the Mila organization.

### Migrate an existing WandB account

!!! warning "Add your Mila email first to avoid a duplicate account"
    To avoid creating a duplicate account: add your `@mila.quebec` address to
    the existing WandB account and make it the primary email **before**
    following the steps below. See the WandB documentation on [managing email
    addresses](https://docs.wandb.ai/guides/app/settings-page/emails). Then log
    out from WandB before proceeding.

1. Go to [wandb.ai](https://wandb.ai) and click **Sign in**.
2. Enter your `@mila.quebec` email address. The password field will disappear
   once a recognized SSO domain is detected.
3. Click **Log in** — the browser will redirect you to the Mila SSO page.
4. Select the **mila.quebec** identity provider. WandB will offer to link the
   existing account to the Mila organization.

### Create a new account

Follow the same SSO steps above. At the account creation prompt, select
**Professional**.

??? question "Which account type to select?"
    Select **Professional** at the account creation prompt. This unlocks team
    features required for the Mila organization. The Mila IT team manages
    organization-level billing, so no personal plan upgrade is required.

## Authenticate the CLI on the cluster

Most WandB Python API calls require a valid API key stored in the environment.

### Log in interactively

Install WandB as a tool on a login node, then authenticate:

```bash
uv tool install --upgrade wandb
wandb login
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
Resolved 20 packages in 371ms
Prepared 20 packages in 1.87s
Installed 20 packages in 2.16s
 + annotated-types==0.7.0
 + certifi==2026.2.25
 [...]
 + urllib3==2.6.3
 + wandb==0.25.1
Installed 2 executables: wandb, wb
wandb: Logging into https://api.wandb.ai.
wandb: Create a new API key at: https://wandb.ai/authorize?ref=models
wandb: Store your API key securely and do not share it.
wandb: Paste your API key and hit enter:
wandb: Appending key for api.wandb.ai to your netrc file: /home/mila/u/username/.netrc
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
```
</div>

!!! tip
    `uv tool install --upgrade wandb` is a one-time step per cluster. The API
    key is stored in `~/.netrc` after `wandb login`; subsequent jobs do not need
    to re-run `wandb login` unless the key is rotated.

### Offline mode

On clusters without outbound internet access on compute nodes, run in offline
mode and sync after the job completes:

```bash
export WANDB_MODE=offline
srun uv run --offline python main.py
```

After the job finishes, sync the run from the login node:

```bash
wandb sync --sync-all
```

!!! tip
    On DRAC clusters, loading the `httpproxy` module before the `srun` line is
    an alternative to offline mode:
    ```bash
    module load httpproxy
    srun uv run python main.py
    ```

## Initialize and log a training run

Every WandB run starts with `wandb.init()`. Call `wandb.finish()` at the end of
the script — it is called automatically at exit, but an explicit call is safer
in Slurm jobs.

### Initialize a run

```python
import os
import wandb

wandb.init(
    project="wandb-example",                # (1)!
    name=os.environ.get("SLURM_JOB_ID"),    # (2)!
    id=os.environ.get("SLURM_JOB_ID"),      # (3)!
    resume="allow",                         # (4)!
    config=vars(args)
    | {f"env/{k}": v for k, v in os.environ.items() if k.startswith("SLURM")},  # (5)!
)
```
{ .annotate }

1. Groups runs in the WandB UI. Use one project per research question.
2. Human-readable display name shown in the WandB Runs table.
3. Sets the unique run ID. `resume="allow"` uses this to find and resume the run
   if it was preempted. Setting it to the Slurm job ID also links the run to its
   log file (`slurm-<JOBID>.out`).
4. Creates a new run if the ID does not exist, or resumes it if it does. Useful
   when combined with checkpointing to recover from preemption.
5. Pass the full `argparse` namespace and SLURM environment variables for easier
   debugging. Every key becomes a searchable, filterable column in the WandB
   Runs table under **Config**.

### Log metrics

Call `wandb.log()` inside the batch loop to record training metrics at each
step:

```python
wandb.log(
    {
        "train/accuracy": accuracy,
        "train/loss": loss
    }
)
```

Log validation metrics once per epoch, after the validation loop:

```python
wandb.log(
    {
        "val/accuracy": val_accuracy,
        "val/loss": val_loss,
        "epoch": epoch
    }
)
```

!!! tip
    Prefix metric names with `train/` and `val/`. WandB groups metrics with
    matching prefixes automatically in the Charts panel.

!!! tip "Complete example"
    See the [WandB setup
    example](../examples/good_practices/wandb_setup/index.md) for a complete
    single-GPU training script with WandB logging integrated.

## Organize runs

Three arguments help keep runs organized as a project grows. `name=` and `tags=`
make individual runs easy to identify and filter in the Runs table, while
`group=` clusters related runs — such as multi-seed runs or ablations — under a
single expandable row.

```python
wandb.init(
    project="wandb-example",
    name=os.environ.get("SLURM_JOB_ID"),
    id=os.environ.get("SLURM_JOB_ID"),
    group=os.environ.get("SLURM_ARRAY_JOB_ID"),     # (1)!
    resume="allow",
    tags=["example", "resnet18"],                   # (2)!
    config=vars(args)
    | {f"env/{k}": v for k, v in os.environ.items() if k.startswith("SLURM")},
)
```
{ .annotate }

1. Using `SLURM_ARRAY_JOB_ID` automatically as the group clusters all jobs into
   a single expandable row.
2. Labels runs for filtering in the Runs table.

## Diagnose training bottlenecks

WandB records GPU utilization, CPU usage, and memory under the **System** tab of
every run automatically, no extra code is required. These metrics are the first
place to check when a training job is slower than expected.

### Read system metrics

Open a run in the WandB UI and select the **System** tab. The **GPU
Utilization** chart shows the fraction of time the GPU spent on active compute
during each sampling interval.

Two patterns indicate different root causes:

- **Sustained utilization near 100%** — the job is compute-bound. The GPU is the
  bottleneck; this is the expected state for well-configured training.
- **Low or oscillating utilization** — the GPU idles while waiting for the next
  batch. The data pipeline cannot deliver batches fast enough; the job is
  I/O-bound.

!!! tip
    Common fixes for an I/O bottleneck: increase `num_workers` in the
    `DataLoader`, enable `pin_memory=True`, or copy the dataset to
    `$SLURM_TMPDIR` before the job starts.

## Full job scripts

The [`wandb_setup` example](../examples/good_practices/wandb_setup/index.md)
provides a complete job script and training script with data staging and WandB
integration:

=== "job.sh"
    ```bash title="job.sh"
    --8<-- "docs/examples/good_practices/wandb_setup/job.sh"
    ```

=== "main.py"
    ```python title="main.py"
    --8<-- "docs/examples/good_practices/wandb_setup/main.py"
    ```

=== "pyproject.toml"

    ```toml title="pyproject.toml"
    --8<-- "docs/examples/good_practices/wandb_setup/pyproject.toml"
    ```

!!! tip "Example run"
    Running these scripts produces a run like this
    [wandb-example run](https://wandb.ai/lebrice/wandb-example).

---

## Key concepts

`wandb.init()`
:   Starts a WandB run. The most commonly used arguments are `project`, `name`,
    `id`, `group`, `config` and `resume`.

`config=` (in `wandb.init()`)
:   Stores a dictionary of hyperparameters alongside the run. Each key becomes a
    searchable, filterable column in the WandB Runs table. Pass
    `config=vars(args) | {f"env/{k}": v for k, v in os.environ.items() if
    k.startswith("SLURM")}` to capture the full `argparse` namespace and Slurm
    environment variables for easier debugging.

`group=` (in `wandb.init()`)
:   Groups related runs under a single expandable row in the WandB Runs table.
    Useful for multi-seed runs or ablations. Pass `SLURM_ARRAY_JOB_ID` to group
    all tasks in a job array automatically.

`wandb.log()`
:   Records a dictionary of metric values at the current step. Call once per
    iteration or epoch.

`WANDB_MODE`
:   Controls logging mode. Set to `offline` on clusters without outbound internet
    access. Sync with `wandb sync --sync-all` after the job completes.

System metrics
:   GPU utilization, CPU usage, and memory stats collected automatically by WandB
    during a run. Visible under the **System** tab of a run in the WandB UI.

`perf/` prefix
:   Convention for logging performance timing metrics (e.g., `perf/data_load_s`,
    `perf/compute_s`) separately from training metrics to support bottleneck
    diagnosis.
