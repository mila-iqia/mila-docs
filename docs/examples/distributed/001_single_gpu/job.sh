#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3

# Creating the environment for the first time:
# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
#     pytorch-cuda=11.6 -c pytorch -c nvidia
# Other conda packages:
# conda install -y -n pytorch -c conda-forge rich

# Activate pre-existing environment.
conda activate pytorch


# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp -rt $SLURM_TMPDIR/data/ /network/datasets/cifar10.var/cifar10_torchvision/*

# Execute Python script
python main.py
