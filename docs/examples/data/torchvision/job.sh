#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -o errexit


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
#     pytorch-cuda=11.7 scipy -c pytorch -c nvidia
# Other conda packages:
# conda install -y -n pytorch -c conda-forge rich tqdm

# Activate pre-existing environment.
conda activate pytorch


# Prepare data for training
mkdir -p "$SLURM_TMPDIR/data"

# If SLURM_JOB_CPUS_PER_NODE is defined and not empty, use the value of
# SLURM_JOB_CPUS_PER_NODE. Else, use 16 workers to prepare data
: ${_DATA_PREP_WORKERS:=${SLURM_JOB_CPUS_PER_NODE:-16}}

# Copy the dataset to $SLURM_TMPDIR so it is close to the GPUs for
# faster training
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
	time -p bash data.sh "/network/datasets/inat" ${_DATA_PREP_WORKERS}


# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

# Execute Python script
srun python main.py
