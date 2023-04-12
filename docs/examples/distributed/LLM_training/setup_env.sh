#!/bin/bash
set -o errexit

# BUG: If a conda environment is already activated, this fails with `ModuleNotFoundError: No module named 'datasets'`

# Install miniconda if 'conda' is not available
function install_miniconda {
    wget "https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" -O $HOME/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
    echo "32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787  $HOME/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" | sha256sum -c -
    bash $HOME/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -u -p "$CONDA_INSTALL_PREFIX"
}

CONDA_INSTALL_PREFIX="$HOME/miniconda3"
export PATH="$PATH:$CONDA_INSTALL_PREFIX/condabin"

which conda >/dev/null || module load anaconda/3 || install_miniconda
module load cuda/11.7

(conda activate base 2>/dev/null) || eval "$(conda shell.bash hook)"

# NOTE: Use a temporary directory if you want to re-create the environment from scratch each time.
# CONDA_ENV_PREFIX=$SLURM_TMPDIR/env
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:=$HOME/conda/llm_training}

echo "Using conda environment at $CONDA_ENV_PREFIX"

# NOTE: Use a temporary directory if you want to re-create the environment from scratch each time.
# CONDA_ENV_PREFIX=$SLURM_TMPDIR/env
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:=$HOME/conda/llm_training}
PACK_ENV=$(basename $CONDA_ENV_PREFIX).tar.gz


if [ ! -d $CONDA_ENV_PREFIX ] || [ ! -f "$PACK_ENV" ]; then
    # Create a conda environment and use the libmamba solver:
    conda create -y -p $CONDA_ENV_PREFIX python=3.9 conda conda-libmamba-solver conda-pack -c conda-forge
    conda activate $CONDA_ENV_PREFIX
    while read f
    do
            if [[ -e "$f" ]]
            then
                    export CONDA_EXE="$f"
                    break
            fi
    done < <(hash -r; which -a conda)

    # Install pytorch:
    conda install --solver=libmamba -y pytorch torchvision torchaudio pytorch-cuda=11.7 transformers datasets evaluate accelerate rich simple-parsing wandb -c pytorch -c nvidia
    # Install other conda packages:
    # conda install -y rich -c conda-forge
    # Install other pip packages:
    pip install "deepspeed>=0.8.2"

    # Pack the environment into an archive using conda-pack.
    conda pack -p $CONDA_ENV_PREFIX
elif [ -z "$SLURM_TMPDIR" ]
then
    conda activate $CONDA_ENV_PREFIX
else
    # Extract the conda env archive to $SLURM_TMPDIR
    if [ ! -d "$SLURM_TMPDIR"/env ]
    then
        echo unpacking tar
        mkdir -p "$SLURM_TMPDIR"/env
        tar -xzf llm_training.tar.gz -C "$SLURM_TMPDIR"/env
    fi
    echo source activate
    source "$SLURM_TMPDIR"/env/bin/activate
    echo conda-unpack
    conda-unpack
fi

# Download dataset
export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub

python3 -c "import datasets ; datasets.load_dataset('wikitext', 'wikitext-103-v1')"

function download_model {
    python3 -c "import transformers ; transformers.AutoConfig.from_pretrained('$1')"
    python3 -c "import transformers ; transformers.AutoTokenizer.from_pretrained('$1', use_fast=True)"
}

# for model_name in "facebook/opt-125m" "facebook/opt-350m" "facebook/opt-1.3b" "facebook/opt-2.7b" "facebook/opt-6.7b" "facebook/opt-13b" "facebook/opt-30b" "facebook/opt-66b" "bigscience/bloom"
# do
#     echo "Downloading model and tokenizer $model_name if needed."
#     download_model $model_name
# done

# Load httpproxy last since it blocks access to HF
! module load httpproxy
python3 -m wandb login
