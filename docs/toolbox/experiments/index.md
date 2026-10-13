---
title: Experiments Tracking & Tuning
description: >-
  Tools to track, compare, and tune machine learning experiments.
---

# Experiments Tracking & Tuning

Research projects often involve many runs with different code versions,
datasets, and hyperparameters. These tools help record, compare, and optimize
those experiments, from logging metrics to running large hyperparameter
searches on the clusters.

<div class="grid cards" markdown>

-   [:material-chart-line:{ .lg .middle } __Comet__](comet.md)
    { .card }

    ---
    Track and compare experiments in a web dashboard on Comet.

-   [:material-tune:{ .lg .middle } __Orion__](orion.md)
    { .card }

    ---
    Run hyperparameter searches in parallel across many cluster jobs.

-   [:simple-weightsandbiases:{ .lg .middle } __Weights and Biases (WandB)__](wandb.md)
    { .card }

    ---
    Log metrics, system usage, and artifacts from training runs, and compare
    them in a web dashboard.

-   [:material-chart-line:{ .lg .middle } __milalib__](https://github.com/mila-iqia/milalib)
    { .card }

    ---
    Monitor compute usage of Slurm jobs, and log these metrics.

</div>
