#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Use `uv run --offline` on clusters without internet access on compute nodes.
uv run python main.py
