---
title: Oríon
description: >-
  Run parallel hyperparameter searches on the Mila cluster with Oríon and Slurm
  job arrays.
---

# Oríon

[Oríon](https://orion.readthedocs.io/en/stable/) is an asynchronous framework
for black-box function optimization developed at Mila. It serves as a
meta-optimizer for machine learning models and training, as well as a flexible
experimentation platform for large-scale asynchronous optimization procedures.

!!! note
    Network Architecture Search (NAS) is a subset of hyperparameter
    optimization (HPO) in which the hyperparameters define the network. The
    same technique applies to both.

## Set up

On a login node, add Oríon to the project environment :

```bash
uv add orion
```

## Use in a Slurm job

Hyperparameter optimization is well suited to parallelization: each trial
(unique set of hyperparameters) is independent of the others. Launch as many
jobs as possible, each trying a different set of hyperparameters and reporting
its results back to a shared location (the Oríon database). The steps below
go in the job script.

### Configure Oríon

Oríon saves all the results of its optimization process in a database. By
default, it uses `pickleddb`, a local database stored on a shared file system.
Specify the location of the database and the name of the experiment.
Optionally, configure workers to run in parallel to maximize resource usage.

```bash
cat > $ORION_CONFIG <<- EOM
    experiment:
        name: ${EXPERIMENT_NAME}
        algorithms:
            hyperband:
                seed: null
        max_broken: 10

    worker:
        n_workers: $SLURM_CPUS_PER_GPU
        pool_size: 0
        executor: joblib
        heartbeat: 120
        max_broken: 10
        idle_timeout: 60

    storage:
        database:
            host: $SCRATCH/${EXPERIMENT_NAME}_orion.pkl
            type: pickleddb
EOM
```

!!! warning
    Move the Oríon database out of `$SCRATCH` once the experiment is done.
    `$SCRATCH` is not backed up and files on it are purged after 90 days.

### Define the search space

Define the search space that Oríon explores. The search space depends on the
model, the optimizer, and other parts of the training. The example below
covers a common set of hyperparameters.

```bash
# Define the hyperparameter search space
cat > $SPACE_CONFIG <<- EOM
    {
        "epochs": "orion~fidelity(1, 100, base=2)",
        "lr": "orion~loguniform(1e-5, 1.0)",
        "weight_decay": "orion~loguniform(1e-10, 1e-3)",
        "momentum": "orion~loguniform(0.9, 1.0)"
    }
EOM
```

### Run `orion hunt`

Once Oríon is configured and has a search space, run `orion hunt`. It starts
the training with a different set of hyperparameters for each worker.

```bash
uv run orion hunt --config $ORION_CONFIG \
    python train.py --config $SPACE_CONFIG --arg1 value1
```

### Launch a job array

Estimate the resources the training requires and adapt the example below
accordingly. The example uses a
[job array](https://slurm.schedmd.com/job_array.html) of 100 jobs, each with
one GPU and 2 CPU cores. Each job runs one Oríon worker per CPU core, and all
workers of a job share the same GPU to maximize its utilization. Up to 200 sets
of hyperparameters are therefore evaluated in parallel across 100 GPUs.

A job array launches the same job script `n` times. Oríon generates different
hyperparameters for each worker.

```bash
sbatch --array=1-100 --gres=gpu:1 --cpus-per-gpu=2 --mem-per-gpu=16G \
    hpo_launcher.sh train.py
```

## Example

The `hpo_launcher.sh` script submitted above combines the configuration,
search space, and `orion hunt` steps:

```bash title="hpo_launcher.sh"
--8<-- "docs/examples/hpo_launcher.sh"
```

For a step-by-step example that adapts a PyTorch training script to Oríon,
see [Hyperparameter Optimization with Oríon](../../examples/good_practices/hpo_with_orion/index.md).

---

## Resources

<div class="grid cards" markdown>

-   [:material-book-open-variant:{ .lg .middle } __Oríon documentation__](https://orion.readthedocs.io/en/stable/)
    { .card }

    ---
    Official user guide, algorithms, and configuration reference.

-   [:simple-github:{ .lg .middle } __Oríon on GitHub__](https://github.com/Epistimio/orion)
    { .card }

    ---
    Source code, issues, and release notes.

</div>
