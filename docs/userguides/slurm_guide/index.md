---
title: Launch jobs
description: Learn the basics of running jobs on the cluster with Slurm.
---

# Launch jobs

## Before you begin

<div class="grid cards" markdown>
    

-   [:material-console-line:{ .lg .middle } __Write code__](../code_examples.md)
    { .card }

    ---
    Start from ready-to-run minimal examples or from the Research Project
    Template to write code that runs on the Mila cluster.

&nbsp;

</div>


## Launch jobs


This section introduces the core Slurm concepts for new users and walks through
running one or more tasks on the cluster, from a first interactive job to
monitoring, managing and synchronizing tasks across multiple nodes.

!!! tip "Work from VSCode on a compute node"
    These guides connect to the cluster with [VSCode](../../toolbox/VSCode.md)
    through `mila code` or the `mila-cpu` remote, which opens a compute node
    with a file browser for `$SCRATCH` and an integrated terminal for the Slurm
    commands. Set it up in the [Get Started
    guide](../../getting_started/index.md). Every step also lists the equivalent
    `ssh mila` terminal command as an alternative.

<div class="grid cards" markdown>

-   [:material-lightbulb-alert-outline:{ .lg .middle } __Understand Slurm__](basics.md)
    { .card }

    ---
    Discover Slurm jobs, steps and tasks. Run multiple tasks through an
    interactive job, then reproduce the example from a batch script.

-   [:material-monitor-eye:{ .lg .middle } __Submit Jobs across Clusters__](cluv.md)
    { .card }

    ---
    Easily use multiple clusters to submit jobs.

-   [:material-monitor-eye:{ .lg .middle } __Monitor and manage jobs__](monitor_manage.md)
    { .card }

    ---
    Track jobs through the queue, inspect and cancel them, read their output,
    and resolve common failures.

-   [:material-shuffle-variant:{ .lg .middle } __Synchronizing multiple tasks__](tasks_communication.md)
    { .card }

    ---
    An applied example showing how tasks running on different nodes can
    communicate and synchronize their output.

</div>

!!! monitoring "Add monitoring to job scripts"
    A job script can collect resource metrics alongside the training script.
    For example, start `milalib` in the background before the training step
    to record GPU usage in a file:

    ```bash title="job.sh"
    uvx milalib monitor -i 5 -m gpu_util -m sm_occupancy \
        > "milalib-$SLURM_JOB_ID.log" &
    srun uv run python main.py
    ```
    
    See [Monitor and manage jobs](monitor_manage.md) and
    [Monitor and Optimize Experiments](../monitoring.md#prepare-metric-collection-before-running).


---


## Next step

<div class="grid cards" markdown>

-   [:material-trending-up:{ .lg .middle } __Monitor and Optimize Experiments__](../monitoring.md)
    { .card }

    ---
    Understand why metrics matter, plan their collection before a job starts,
    and choose the right tool to optimize both models and resource usage.

-   [:material-server:{ .lg .middle } __Compute Utilization Dashboard__](../compute_utilization_guidelines/index.md)
    { .card }

    ---
    Use the dashboard to identify and reduce wasted GPU resources.
    

-   [:material-newspaper-variant-multiple-outline:{ .lg .middle } __Share results and make research reproducible__](../reproducibility.md)
    { .card }

    ---
    Make cluster experiments reproducible and shareable — environment management, version control, dataset sharing, and research paper distribution.

&nbsp;

</div>