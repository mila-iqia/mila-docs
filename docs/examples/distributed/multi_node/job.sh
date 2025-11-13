#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Exit on error
set -e

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Stage dataset into $SLURM_TMPDIR (only on the first worker of each node)
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c \
   'mkdir -p $SLURM_TMPDIR/data && ln -s /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/'

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Execute Python script in each task (one per GPU)
# Use the `--offline` option of `uv run` on clusters without internet access on compute nodes.
# Using the `--locked` option can help make your experiments easier to reproduce (it forces
# your uv.lock file to be up to date with the dependencies declared in pyproject.toml).
srun uv run python main.py

