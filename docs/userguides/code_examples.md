---
title: Write code - Examples
description: >-
  Start from ready-to-run minimal examples or from the Research Project
  Template to write code that runs on the Mila cluster.
---

# Write Code from Examples

Writing research code from scratch is rarely necessary. Mila provides two
starting points: a collection of **minimal examples** that each illustrate one
concept in isolation, and the **Research Project Template**, a complete
project structure ready to be used for a new research project. Both are
designed to run on the Mila cluster as-is.

## Before you begin

<div class="grid cards" markdown>

-   [:material-run-fast:{ .lg .middle } __Get Started with the Cluster__](../getting_started/index.md)
    { .card }

    ---
    Obtain a Mila account, enable cluster access and MFA, install `uv` and
    `milatools`, configure SSH access and connect to the cluster for the first
    time.

&nbsp;

</div>

## What this page covers

* Choose between the minimal examples and the Research Project Template
* Find the minimal example matching a specific need
* Start a new project from the Research Project Template

!!! monitoring "Monitor & Optimize"
    To set up monitoring, see [Monitoring and Optimize](monitoring.md)

---

## Choose a starting point

| Need                                                      | Starting point                                   |
| --------------------------------------------------------- | ------------------------------------------------ |
| Understand one concept (multi-GPU, checkpointing, etc.)   | [Minimal examples](#minimal-examples)            |
| Adapt an existing script to run on the cluster            | [Minimal examples](#minimal-examples)            |
| Start a new research project with a full, tested codebase | [Research Project Template](#research-project-template) |

## Minimal examples

The [minimal examples](../examples/index.md) are small, self-contained
projects. Each one contains a `job.sh` Slurm script, launched with
`sbatch job.sh`, and a `main.py` Python script. Many examples are displayed as
a difference with respect to a simpler base example, which highlights exactly
what changes are required to add a feature.

<div class="grid cards" markdown>

-   [:material-cog:{ .lg .middle } __Software Setup__](../examples/frameworks/index.md)
    { .card }

    ---
    Set up an environment with PyTorch, JAX or Flash Attention using `uv`.

-   [:material-format-list-bulleted:{ .lg .middle } __Distributed Training__](../examples/distributed/index.md)
    { .card }

    ---
    Scale a training script from a single GPU to multiple GPUs and nodes.

-   [:material-check-decagram:{ .lg .middle } __Good Practices__](../examples/good_practices/index.md)
    { .card }

    ---
    Add checkpointing, experiment tracking, job arrays and hyperparameter
    search.

-   [:material-rocket-launch:{ .lg .middle } __Advanced Examples__](../examples/advanced/index.md)
    { .card }

    ---
    Combine these concepts, for instance to train on ImageNet across multiple
    nodes.

</div>

!!! tip "Read the examples in order"
    Each distributed training example builds on the previous one. Start with
    the [single GPU job](../examples/distributed/single_gpu/index.md) before
    moving on to the multi-GPU and multi-node examples.

!!! monitoring "Add metric logging from the start"
    The [WandB setup example](../examples/good_practices/wandb_setup/index.md)
    shows a training script that already logs metrics. Adding logging while
    writing the code makes the results available in Phase 3. See
    [Monitor and Optimize Experiments](monitoring.md#prepare-metric-collection-before-running).

## Research Project Template

The [Research Project Template](https://mila-iqia.github.io/ResearchTemplate/)
is a starting point for new machine learning research projects. It provides a
project structure that already integrates the tools commonly used at Mila:

* [Hydra](https://hydra.cc/) to configure experiments.
* [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and
  [JAX](https://docs.jax.dev/en/latest/) to write training code.
* [Weights & Biases](https://wandb.ai/) to track experiments.
* [pytest](https://docs.pytest.org/en/stable/) to test the code.

Instead of assembling these components one by one, the template provides them
already configured and working together, so research can start on a tested
codebase. Follow the instructions in the
[Research Project Template documentation](https://mila-iqia.github.io/ResearchTemplate/)
to create a new project from it.

!!! note "Other tools"
    The [Toolbox](../toolbox/index.md) section describes more tools that can
    be used alongside the template or the minimal examples.

---

## Next step

<div class="grid cards" markdown>

-   [:material-server:{ .lg .middle } __Launch jobs__](slurm_guide/index.md)
    { .card }

    ---
    Submit and manage jobs on the cluster with Slurm.

&nbsp;

</div>
