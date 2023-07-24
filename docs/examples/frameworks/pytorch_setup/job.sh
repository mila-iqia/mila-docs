#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3

# Activate the environment:
# NOTE: Use the `make_env.sh` script to create the environment if you haven't already.
conda activate pytorch

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

python main.py
