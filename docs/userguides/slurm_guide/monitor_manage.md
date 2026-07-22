---
title: Monitor and manage jobs
description: Track, inspect, cancel and troubleshoot jobs on the cluster.
---

# Monitor and manage jobs

Once a job is submitted, Slurm places it in a queue and starts it when the
requested resources become available. This guide covers how to follow a job
through the queue, inspect it while it runs and after it finishes, check how
efficiently it used its allocation, cancel it, read its output, and resolve the
failures new users hit most often.

!!! note "Where to run these commands"
    The commands below run in the VSCode integrated terminal on a compute node
    (through `mila code` or the `mila-cpu` remote), or in a login-node terminal
    after `ssh mila`. See [VSCode](../../toolbox/VSCode.md).

## Before you begin

<div class="grid cards" markdown>

-   [:material-lightbulb-alert-outline:{ .lg .middle } __Understand Slurm__](basics.md)
    { .card }

    ---
    Submit interactive and batch jobs, and learn the jobs, steps and tasks
    model.

&nbsp;

</div>

## What this guide covers

* Tracking queued and running jobs
* Inspecting finished jobs and their resource usage
* Cancelling jobs
* Reading job output
* Troubleshooting common failures

---

## Track queued and running jobs

The [`squeue`](https://slurm.schedmd.com/squeue.html) command lists jobs known
to the scheduler. The `--me` flag restricts the output to the current user's
jobs:

```bash
squeue --me
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
JOBID     USER    PARTITION           NAME  ST START_TIME             TIME NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) COMMENT
9321166 user.name long-cpu,lon      job.sh  PD N/A                    0:00     2    4        N/A      2G  (Priority) (null)
```
</div>

The `ST` (state) column reports where a job stands, and the trailing `(REASON)`
column explains why a pending job has not started yet:

| State | Meaning |
| ----- | ------- |
| `PD`  | Pending — waiting in the queue for resources |
| `R`   | Running |
| `CG`  | Completing — finishing and releasing its resources |

For pending jobs, `(Priority)` and `(Resources)` are the most common reasons and
mean the job is waiting its turn. Reasons that point to a problem are covered in
[Troubleshoot common failures](#troubleshoot-common-failures) below.

!!! note
    Each user can keep up to 1000 jobs queued at once. See the
    [Checking job status](../../technical_reference/general_theory/slurm.md#checking-job-status)
    reference for the full list of `squeue` fields and states.

## Inspect a finished job

Once a job finishes, it disappears from `squeue`. The
[`sacct`](https://slurm.schedmd.com/sacct.html) command reports accounting data
for jobs that are still running or have already completed:

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem
```
<div class="result" style="border:None; padding:0" markdown>
``` linenums="0"
JobID             State ExitCode    Elapsed     MaxRSS     ReqMem
------------ ---------- -------- ---------- ---------- ----------
9321166       COMPLETED      0:0   00:00:07                    2G
9321166.bat+  COMPLETED      0:0   00:00:07      1428K
9321166.ext+  COMPLETED      0:0   00:00:07       112K
9321166.0     COMPLETED      0:0   00:00:02      1352K
```
</div>

Each job lists one line per step (here the batch step, the housekeeping
`extern` step and the `srun hostname` step, numbered `0`); per-step fields
such as `MaxRSS` appear on the step lines.

The `--format` flag selects which columns to display. The fields above answer
the questions that matter most after a job ends:

* `State` and `ExitCode` — whether the job succeeded (`COMPLETED`, `0:0`) or
  failed (for example `FAILED`, `TIMEOUT` or `OUT_OF_MEMORY`)
* `Elapsed` — how long the job actually ran
* `MaxRSS` — the peak memory a task used
* `ReqMem` — the memory that was requested

See the [info commands](../../technical_reference/general_theory/slurm.md#checking-job-status)
reference for the complete field list.

## Check resource efficiency

Comparing the resources a job *used* against what it *requested* is the fastest
way to spot waste. Over-requesting memory or time makes a job harder to schedule
and holds resources other users could run on. Comparing `MaxRSS` against
`ReqMem` (from `sacct` above) is a good starting point for CPU and memory jobs.

<div class="grid cards" markdown>

-   [:material-speedometer:{ .lg .middle } __Identifying GPU waste__](../gpu_efficiency.md)
    { .card }

    ---
    Diagnose under-used GPUs and right-size a GPU allocation.

-   [:material-chart-line:{ .lg .middle } __Monitoring__](../../technical_reference/clusters/mila/monitoring.md)
    { .card }

    ---
    Watch live CPU, memory and GPU usage on a compute node with Netdata and
    Grafana.

</div>

## Cancel a job

The [`scancel`](https://slurm.schedmd.com/scancel.html) command stops a job,
whether it is pending or running:

```bash
scancel <JOB_ID>
```

To cancel every job belonging to the current user at once:

```bash
scancel --me
```

Cancelling a job releases its allocation immediately. See the FAQ entry
[How do I cancel a job?](../../help/faq.md#how-do-i-cancel-a-job) for more
options.

## Read job output

By default, a batch job writes both its standard output and standard error to a
file named `slurm-<JOB_ID>.out` in the directory the job was submitted from
(see [Retrieve the results](basics.md#retrieve-the-results)). Change the
destination with the
[`--output`](https://slurm.schedmd.com/sbatch.html#OPT_output) and
[`--error`](https://slurm.schedmd.com/sbatch.html#OPT_error) directives:

```bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
```

The `%x` and `%j` patterns are replaced with the job name and job ID, which
keeps output files from different jobs separate. Interactive jobs launched with
`salloc` print their output directly to the terminal instead.

## Troubleshoot common failures

Most job problems fall into a handful of categories. The table below maps each
symptom to its cause and fix, and links to the reference entry with the full
explanation.

| Symptom | Cause and fix |
| ------- | ------------- |
| Pending with `(Priority)` or `(Resources)` | Normal queueing — the job waits for higher-priority jobs to run or for resources to free up. Request less time or fewer resources to start sooner. See [Understanding the queue](../../technical_reference/general_theory/batch_scheduling.md#understanding-the-queue). |
| Pending with `(QOSMaxJobsPerUserLimit)` | Too many jobs are queued or running under the current limits. Wait for earlier jobs to finish before submitting more. |
| Pending with `(ReqNodeNotAvail)` | The requested nodes or features are unavailable (for example an impossible `--constraint`). Relax the constraint or choose another partition. |
| Killed with an `oom-kill` message | The job exceeded its memory allocation. Request more memory with `--mem`. See [the oom-kill FAQ entry](../../help/faq.md#slurmstepd-error-detected-1-oom-kill-events-in-step-batch-cgroup). |
| Ends in `TIMEOUT` | The job hit its `--time` limit. Request more time, or add checkpointing to resume long runs. See [Checkpointing](../../examples/good_practices/checkpointing/index.md). |
| `Invalid account or account/partition combination` | The requested partition or account is not valid for this user. See [the FAQ entry](../../help/faq.md#unable-to-allocate-resources-invalid-account-or-accountpartition-combination-specified). |
| `--mem and --mem-per-cpu are mutually exclusive` | Only one memory flag can be set. Keep either `--mem` or `--mem-per-cpu`. See [the FAQ entry](../../help/faq.md#srun-error-mem-and-mem-per-cpu-are-mutually-exclusive). |

---

## Key concepts

Job state
:   The stage a job has reached, reported in the `ST` column of `squeue` (for
    example `PD` pending, `R` running) or the `State` column of `sacct` (for
    example `COMPLETED`, `FAILED`, `TIMEOUT`).

Reason code
:   The value in the `(REASON)` column of `squeue` that explains why a pending
    job has not started, such as `Priority`, `Resources` or `ReqNodeNotAvail`.

## Next step

<div class="grid cards" markdown>

-   [:material-shuffle-variant:{ .lg .middle } __Synchronizing multiple tasks__](tasks_communication.md)
    { .card }

    ---
    Synchronize the output of multiple tasks running on different nodes.

&nbsp;

</div>
