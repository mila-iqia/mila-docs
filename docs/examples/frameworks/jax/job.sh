#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:15:00

# Exit on error
set -e

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# To make your code as much reproducible as possible, uncomment the following
# block:
## === Reproducibility ===
## BROKEN: This seams to block the training and nothing happens.
## Be warned that this can make your code slower. See
## https://github.com/jax-ml/jax/issues/13672 for more details.
## export XLA_FLAGS=--xla_gpu_deterministic_ops=true
## === Reproducibility (END) ===

# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
# General-purpose alternatives combining copy and unpack:
#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/

# Execute Python script
# Use `uv run --offline` on clusters without internet access on compute nodes.
srun uv run python main.py
