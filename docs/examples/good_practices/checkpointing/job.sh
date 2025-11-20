#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300 # tells the controller to send SIGTERM to the job 5
                            # min before its time ends to give it a chance for
                            # better cleanup. If you cancel the job manually,
                            # make sure that you specify the signal as TERM like
                            # so `scancel --signal=TERM <jobid>`.
                            # https://dhruveshp.com/blog/2021/signal-propagation-on-slurm/

# Echo time and hostname into log

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "Job has been preempted $SLURM_RESTART_COUNT times."

# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
# General-purpose alternatives combining copy and unpack:
#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/

# Execute Python script
# Use the `--offline` option of `uv run` on clusters without internet access on compute nodes.
# Using the `--locked` option can help make your experiments easier to reproduce (it forces
# your uv.lock file to be up to date with the dependencies declared in pyproject.toml).
# Here we use `exec` to ensure that the signals are received and handled in the Python process.
exec srun uv run python main.py
