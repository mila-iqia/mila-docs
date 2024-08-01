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
module load anaconda/3
module load cuda/11.7

# default values, change if found elsewhere
VENV_DIR="$SLURM_TMPDIR/env"
IMAGENET_DIR=$SLURM_TMPDIR/imagenet 

if [ ! -d "$IMAGENET_DIR" ]; then
  echo "ImageNet dataset not found. Preparing dataset..."
  ./make_imagenet.sh
else
  echo "ImageNet dataset already prepared."
fi

# Check if virtual environment exists, create it if it doesn't
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment not found. Creating it."
    module load python/3.10
    python -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install torch rich tqdm torchvision scipy wandb
else
    echo "Activating pre-existing virtual environment."
    source $VENV_DIR/bin/activate
fi

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

# Execute Python script
python main.py "$@"
