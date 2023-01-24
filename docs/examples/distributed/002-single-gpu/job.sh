#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module purge
module load anaconda/3


# Activate pre-existing environment.
conda activate py38torch113


# Stage dataset into $SLURM_TMPDIR
cp -a /network/datasets/cifar10.var/cifar10_torchvision $SLURM_TMPDIR


# Execute Python script
python main.py
