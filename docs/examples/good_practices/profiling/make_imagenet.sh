#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
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
# conda create -y -n pytorch python=3.9 
# pip install torch rich tqdm torchvision scipy

# Activate pre-existing environment.
conda activate pytorch

# ImageNet setup
echo "Setting up ImageNet directories and creating symlinks..."
mkdir -p $SLURM_TMPDIR/imagenet
ln -s /network/datasets/imagenet/ILSVRC2012_img_train.tar -t $SLURM_TMPDIR/imagenet 
ln -s /network/datasets/imagenet/ILSVRC2012_img_val.tar -t $SLURM_TMPDIR/imagenet
ln -s /network/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz -t $SLURM_TMPDIR/imagenet
echo "Creating ImageNet validation dataset..."
python -c "from torchvision.datasets import ImageNet; ImageNet('$SLURM_TMPDIR/imagenet', split='val')"
echo "Creating ImageNet training dataset..."
mkdir -p $SLURM_TMPDIR/imagenet/train
tar -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar \
     --to-command='mkdir -p $SLURM_TMPDIR/imagenet/train/${TAR_REALNAME%.tar}; \
                    tar -xC $SLURM_TMPDIR/imagenet/train/${TAR_REALNAME%.tar}' \
     -C $SLURM_TMPDIR/imagenet/train
# SLOWER: Obtain ImageNet files using torch directly
#python -c "from torchvision.datasets import ImageNet; ImageNet('$SLURM_TMPDIR/imagenet', split='train')"