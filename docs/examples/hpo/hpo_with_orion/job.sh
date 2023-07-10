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
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.7

# Creating the environment for the first time:
# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
#     pytorch-cuda=11.7 -c pytorch -c nvidia
# Other conda packages:
# conda install -y -n pytorch -c conda-forge rich tqdm
# Orion package:
# pip install orion

# Activate pre-existing environment.
conda activate pytorch


# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
# General-purpose alternatives combining copy and unpack:
#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/


# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

# =============
# Execute Orion
# =============

# Specify an experiment name with `-n`,
# which could be reused to display results (see section "Running example" below)

# Specify max trials (here 10) to prevent a too-long run.

# Then you can specify a search space for each `main.py`'s script parameter
# you want to optimize. Here we optimize only the learning rate.

orion hunt -n orion-example --exp-max-trials 10 python main.py --learning-rate~'loguniform(1e-5, 1.0)'
