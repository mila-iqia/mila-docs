#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Exit on error
set -e

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# To make your code as much reproducible as possible with
# `torch.use_deterministic_algorithms(True)`, uncomment the following block:
## === Reproducibility ===
## Be warned that this can make your code slower. See
## https://pytorch.org/docs/stable/notes/randomness.html#cublas-and-cudnn-deterministic-operations
## for more details.
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
## === Reproducibility (END) ===

# Stage dataset into $SLURM_TMPDIR (only on the first worker of each node)
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c \
   'mkdir -p $SLURM_TMPDIR/data && cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/'

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Execute Python script in each task (one per GPU)
# Use the `--offline` option of `uv run` on clusters without internet access on compute nodes.
# Using the `--locked` option can help make your experiments easier to reproduce (it forces
# your uv.lock file to be up to date with the dependencies declared in pyproject.toml).
srun uv run python main.py

