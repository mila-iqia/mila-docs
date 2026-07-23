---
title: Partitions
description: Slurm partitions available on the Mila cluster, their resource limits, and how to choose between them.
---

# Partitions

Partitions are Slurm's mechanism for dividing the cluster's limited GPU
resources fairly among jobs. Each partition applies a different combination of
resource limits, maximum run time, and priority. Use the Slurm flag
`--partition`/`-p` to select which partition a job runs in.

## Partition details

Each job assigned with a priority can preempt jobs with a lower priority:
`unkillable > main > long`.  Once preempted, the job is killed without notice
and is automatically re-queued on the same partition until resources are available.
To leverage a different preemption mechanism, see the 
[Handling preemption](../../../general_theory/multigpu/#handling-preemption) page.

| Partition          | Max Resource Usage        | Max Time    | Note                                                       |
| -------------------| ------------------------- | ----------- | ---------------------------------------------------------- |
| `unkillable`       | 6  CPUs, mem=32G,  1 GPU  | 2 days      |                                                            |
| `unkillable-cpu`   | 2  CPUs, mem=16G          | 2 days      | CPU-only jobs                                              |
| `short-unkillable` | mem=1000G, 4 GPUs         | **3 hours** | Large but short jobs. <br> Restricted to 4-GPU nodes only  |
| `main`             | 8  CPUs, mem=48G,  2 GPUs | 5 days      |                                                            |
| `main-cpu`         | 8  CPUs, mem=64G          | 5 days      | CPU-only jobs                                              |
| `long`             | No limit of resources     | 7 days      |                                                            |
| `long-cpu`         | No limit of resources     | 7 days      | CPU-only jobs                                              |


???+ warning "Important: H100 GPUs Partition Restrictions"

    H100 GPUs are **ONLY** available in the `short-unkillable` partition.
    
    The `short-unkillable` partition is restricted to 4-GPU nodes only,
    specifically:
    
    - **cn-g nodes**: A100 80GB GPUs (4 GPUs per node)
    - **cn-l nodes**: L40S GPUs (4 GPUs per node)

    As an exception, it also contains the H100 nodes:

    - **cn-n nodes**: H100 GPUs (8 GPUs per node, but only 4 can be used per job)
    
    For a complete list of node specifications and GPU details, see [Node
    profile description](../nodes).

!!! note
    *As a convenience*, requesting the `unkillable`, `main`, or `long`
    partition for a CPU-only job automatically translates it to the `-cpu`
    equivalent.

## When to use each partition and GPU availability

The following table provides a quick reference guide for choosing partitions and
understanding GPU availability:

| Partition          | When to use                                                          | Available GPUs                        |
|--------------------|----------------------------------------------------------------------|---------------------------------------|
| `unkillable`       | High-priority jobs that cannot be interrupted.                       | All GPU types                         |
| `short-unkillable` | Large short jobs that need high priority and cannot be interrupted.  | **Restricted to 4-GPU nodes only**    |
| `main`             | Standard priority jobs with moderate runtime needs.                  | All GPU types                         |
| `long`             | Long-running jobs that can tolerate preemption.                      | All GPU types **except H100**         |
| `*-cpu`            | CPU-only jobs (no GPU required).                                     | N/A (CPU-only nodes)                  |