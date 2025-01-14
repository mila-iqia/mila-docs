#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=main

# NOTE: Run this either with `sbatch make_env.sh` or within an interactive job with `salloc`:
# salloc --gres=gpu:1 --cpus-per-task=1 --mem=16G --time=00:30:00

# Exit on error
set -e

module --quiet purge
module load anaconda/3
module load cuda/11.7

ENV_PATH="$SCRATCH/conda/pytorch_orion"

# Create the environment:
conda create --yes --prefix $ENV_PATH python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.7 --channel pytorch --channel nvidia
# Install as many packages as possible with Conda:
conda install --yes --prefix $ENV_PATH tqdm rich --channel conda-forge
# conda install --yes --prefix $ENV_PATH orion --channel epistimio  # NOTE: Unfortunately this doesn't work atm: https://github.com/Epistimio/orion/issues/1111
# Activate the environment:
conda activate $ENV_PATH
# Install the rest of the dependencies with pip:
pip install orion
conda env export --no-builds --from-history --file environment.yaml
