---
title: Comet
description: >-
  Track and compare machine learning experiments from cluster jobs in the
  Mila organization on Comet.
---

# Comet

[Comet](https://www.comet.com/docs/v2/) is an experiment tracking platform
that logs metrics, hyperparameters, code, and system usage from training runs,
and compares them in a web dashboard. Mila hosts its own Comet instance at
[comet.mila.quebec](https://comet.mila.quebec/), where experiments are shared
within the Mila organization (`mila-org`).

## Before you begin

!!! success "Requirements"
    - **Mila organization access** Students supervised by core professors are
      eligible. Request access by writing to
      [it-support@mila.quebec](mailto:it-support@mila.quebec).
    - **Python environment** A project environment on the cluster where the
      `comet_ml` package can be installed.

---

## Get access

Sign in at [https://comet.mila.quebec/](https://comet.mila.quebec/) to access
`mila-org`. The account in `mila-org` is independent from any personal Comet
account. On first login, one of the following applies:

=== "No account, or account with another email"

    Comet creates a new account in `mila-org`. This applies when no Comet
    account exists, or when the existing account uses an email address other
    than `@mila.quebec`.

=== "Account with a @mila.quebec email"

    Comet links the existing account to `mila-org`.

## Set up

Install the Comet Python SDK in the project environment:

```bash
uv add comet_ml
```

Copy the API key from the Comet web interface (**Account settings** →
**API Keys**), then store it in the Comet configuration file on the cluster:

```bash
comet login
```

The key is saved to `~/.comet.config`. Point the SDK to the Mila Comet
instance instead of the public `comet.com` service:

<!-- TODO: confirm the exact URL override for the Mila self-hosted instance -->
```bash
export COMET_URL_OVERRIDE=https://comet.mila.quebec/clientlib/
```

!!! warning
    Never commit the API key or `~/.comet.config` to a Git repository.

## Use in a Slurm job

Create the experiment at the start of the training script; Comet reads the
API key and URL from the configuration file and environment:

```python
import comet_ml

experiment = comet_ml.start(project_name="my-project", workspace="mila-org")
experiment.log_parameters({"lr": 1e-3, "batch_size": 64})

for epoch in range(num_epochs):
    ...
    experiment.log_metric("train/loss", loss, epoch=epoch)

experiment.end()
```

Submit the script from a job, not from a login node:

=== "Mila cluster"

    Compute nodes on the Mila cluster have internet access, so runs log
    directly to Comet.

    ```bash
    #!/bin/bash
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    #SBATCH --time=1:00:00

    export COMET_URL_OVERRIDE=https://comet.mila.quebec/clientlib/
    uv run python main.py
    ```

=== "DRAC clusters"

    Compute nodes on most DRAC clusters have no internet access. Load the
    `httpproxy` module to reach Comet servers, or log offline and upload
    the results afterwards.

    ```bash
    #!/bin/bash
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    #SBATCH --time=1:00:00

    module load httpproxy
    export COMET_URL_OVERRIDE=https://comet.mila.quebec/clientlib/
    uv run python main.py
    ```

    <!-- TODO: confirm httpproxy allows comet.mila.quebec, not only
         comet.com -->

    For offline logging, start the experiment with `online=False`, then
    upload the archive from a login node once the job ends:

    ```bash
    comet upload <path/to/offline/experiment>.zip
    ```

<!-- TODO: add an "## Example" section once a Comet example exists under
     docs/examples/ (see good_practices/wandb_setup for the WandB equivalent).
-->

---

## Resources

<div class="grid cards" markdown>

-   [:material-book-open-variant:{ .lg .middle } __Comet documentation__](https://www.comet.com/docs/v2/)
    { .card }

    ---
    Official Comet user guide.

-   [:material-server:{ .lg .middle } __Using CometML and WandB on DRAC clusters__](../../technical_reference/clusters/drac/index.md#using-cometml-and-wandb)
    { .card }

    ---
    Reach Comet from DRAC compute nodes with httpproxy.

</div>
