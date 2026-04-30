# Checkpointing

## Prerequisites

Checkpointing is the process of periodically saving the full state of a training run so that it
can be resumed after an interruption. Without checkpointing, any interruption means starting from scratch
and wasting compute and time. This is especially useful to manage long-running jobs that can then be split
into more robust and smaller ones

A complete checkpoint generally needs more than just the model weights. It also includes the optimizer
state, the current epoch number and the states of random number generator (for Python, NumPy, and PyTorch, etc.). 
Saving all of these states ensures that a resumed run produces exactly the same result as an uninterrupted one.

You can also save multiple intermediate checkpoints for analysis or to keep different restart points. But be mindful of storage space, as checkpoints can consume a lot of it, especially if your model and optimizer states are large. Make sure to monitor your storage usage and clean up old checkpoints as needed. 

!!! warning
    You can use the `$SCRATCH` storage for checkpoints, but be aware that it is not backed up and may be cleared periodically, so it's not suitable for long-term storage. After the end of the training, you should always keep a copy of the most valuable checkpoints (e.g., the best model or the latest checkpoint) in a more permanent storage.

Checkpointing enables you to:

* Resume training from the last saved checkpoint rather than from scratch after a preemption or
  time-limit, avoiding wasted compute.
* Request shorter wall-clock time allocations, which are easier to schedule on the cluster.
* Access preemptible partitions where jobs get high-priority resources but may be interrupted.

Make sure to read the following sections of the documentation before using this
example:

* [PyTorch Setup](../../frameworks/pytorch_setup/index.md)
* [Single GPU](../../distributed/single_gpu/index.md)

## Example

The full source code for this example is available on [the mila-docs GitHub repository.](https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/good_practices/checkpointing)

**job.sh**

Two SBATCH directives are added compared to the baseline:

* `#SBATCH --requeue`: automatically requeues the job when it is preempted by the scheduler.
* `#SBATCH --signal=B:TERM@300`: sends `SIGTERM` to the job 5 minutes before its time limit
  expires, giving the Python process time to exit cleanly.


```diff
--8<-- "docs/examples/good_practices/checkpointing/job.sh.diff"
```

**pyproject.toml**

```toml
--8<-- "docs/examples/good_practices/checkpointing/pyproject.toml"
```

**main.py**

The main additions are mainly into three parts:

**Checkpoint saving :** At the end of each epoch, `save_checkpoint` writes a `RunState`
dictionary to `$SCRATCH/checkpointing_example/<job_id>/checkpoints/checkpoint.pth`. The
`RunState` includes the model weights, optimizer state, current epoch, best validation accuracy,
and all random number generator states.

**Checkpoint loading :** Before the training begins, `load_checkpoint` checks whether a checkpoint
file already exists at the expected path. If it does, the model, optimizer and all random states are
restored, the epoch counter starts from the next epoch. Because SLURM preserves the job ID across requeues,
the restarted job automatically finds the checkpoint left by the previous run.

**Signal handling :** The `signal_handler` function is registered for `SIGTERM` (preemption or
manual cancellation) and `SIGUSR1` (time-limit warning sent by `--signal=B:TERM@300`). It logs
the received signal and exits with code 0 so that SLURM does not mark the job as failed. Since
the last complete checkpoint was saved at the end of the previous epoch, at most one epoch of
progress is lost when training is interrupted.

```diff
--8<-- "docs/examples/good_practices/checkpointing/main.py.diff"
```

## Running this example

Submit the job script with:
```bash
$ sbatch job.sh
```

To cancel the job manually while allowing it to be requeued, send `SIGTERM` rather than the
default `SIGKILL`:
```bash
$ scancel --signal=TERM <jobid>
```

To requeue a job that is still in the queue:
```bash
$ scontrol requeue <jobid>
```

!!! warning
    A job cancelled with plain `scancel` (no `--signal` flag) is marked as `FAILED` by SLURM
    and cannot be requeued. Always pass `--signal=TERM` when cancelling a job you intend to
    resume.
