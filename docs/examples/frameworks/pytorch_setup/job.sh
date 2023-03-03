#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=unkillable

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3

# Creating the environment for the first time:
# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
#     pytorch-cuda=11.6 -c pytorch -c nvidia

# Activate the environment:
conda activate pytorch

python main.py
