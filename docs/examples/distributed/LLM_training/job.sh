#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=a100:1
#SBATCH --mem=512G
#SBATCH --time=01:00:00

set -e  # exit on error.


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.7

# CONDA_ENV_PREFIX=$SLURM_TMPDIR/env
CONDA_ENV_PREFIX=$SCRATCH/conda/llm_training

if [ ! -d $CONDA_ENV_PREFIX ]; then
    # Create a conda environment and use the libmamba solver:
    conda create -y -p $CONDA_ENV_PREFIX python=3.9 conda conda-libmamba-solver -c conda-forge
    conda activate $CONDA_ENV_PREFIX
    export CONDA_EXE="$(hash -r; which conda)"
    conda config --set solver libmamba

    # Install the pytorch dependencies:
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

    # Install other conda packages:
    # conda install -y rich -c conda-forge

    # Install other pip packages:
    pip install rich transformers datasets evaluate accelerate deepspeed
else
    conda activate $CONDA_ENV_PREFIX
fi

set -x  # print commands.

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:=4}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# export CPATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/include/

# ACCELERATE_CONFIG="gabriele_config.yaml"
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:="gabriele_config.yaml"}

# TODO: Load the dataset in-memory:
# export HF_DATASETS_IN_MEMORY_MAX_SIZE=$SLURM_MEM_PER_NODE

# IDEA: Copy the dataset to the SLURM_TMPDIR of each node:
# (Not sure if this is actually useful).
# srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c 'cp -a $SCRATCH/cache/huggingface/datasets $SLURM_TMPDIR/datasets'
# export HF_DATASETS_CACHE="$SLURM_TMPDIR/datasets"

output_dir=$SCRATCH/logs/llm_training/$SLURM_JOB_ID
mkdir -p $output_dir

# TODO: This should be run once per node with `srun --ntasks-per-node=1 bash -c '...'`
srun --tasks-per-node=1 \
    accelerate launch \
        --config_file=$ACCELERATE_CONFIG \
        --machine_rank=$SLURM_NODEID \
        --num_cpu_threads_per_process=$SLURM_CPUS_PER_TASK \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --num_processes=$SLURM_NTASKS_PER_NODE \
        deepspeed_with_config_support.py \
        --config_name=facebook/opt-2.7b --tokenizer_name=facebook/opt-2.7b \
        --dataset_name=wikitext --dataset_config_name wikitext-103-v1 \
        --per_device_train_batch_size=1 --max_train_steps=10 --output_dir=$output_dir
