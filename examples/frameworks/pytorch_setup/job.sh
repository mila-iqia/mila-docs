#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=l40s:1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:15:00

# Exit on error
set -e

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Execute Python script
# Use `uv run --offline` on clusters without internet access on compute nodes.
uv run python main.py
