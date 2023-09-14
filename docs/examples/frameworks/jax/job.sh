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

# Creating the environment for the first time:
# conda create -y -n jax_ex -c "nvidia/label/cuda-11.8.0" cuda python=3.9 virtualenv pip
# conda activate jax_ex
# Install Jax using `pip`
# *Please note* that as soon as you install packages from `pip install`, you
# should not install any more packages using `conda install`
# pip install --upgrade "jax[cuda11_pip]" \
#    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Other pip packages:
# pip install pillow optax rich torch torchvision flax tqdm

# Activate the environment:
conda activate jax_ex


# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
# General-purpose alternatives combining copy and unpack:
#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/


# Execute Python script
python main.py
