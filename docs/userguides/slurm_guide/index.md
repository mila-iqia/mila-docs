---
title: Launch jobs
description: Learn the basics of running jobs on the cluster with Slurm.
---

# Launch jobs

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
