#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Exit on error
set -e

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp --update /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
# General-purpose alternatives combining copy and unpack:
#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# Execute Python script in each task (one per GPU)
# Use the `--offline` option of `uv run` on clusters without internet access on compute nodes.
# Using the `--locked` option can help make your experiments easier to reproduce (it forces
# your uv.lock file to be up to date with the dependencies declared in pyproject.toml).
srun uv run python main.py
