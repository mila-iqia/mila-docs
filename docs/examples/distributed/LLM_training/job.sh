#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=a100:2
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --job-name=llm_training
#SBATCH --output=logs/slurm-%j.out

set -e  # exit on error.

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.7

# NOTE: Use a temporary directory if you want to re-create the environment from scratch each time.
# CONDA_ENV_PREFIX=$SLURM_TMPDIR/env
CONDA_ENV_PREFIX=$SCRATCH/conda/llm_training



if [ ! -d $CONDA_ENV_PREFIX ]; then
    # Create a conda environment and use the libmamba solver:
    conda create -y -p $CONDA_ENV_PREFIX python=3.9 conda conda-libmamba-solver -c conda-forge
    conda activate $CONDA_ENV_PREFIX
    export CONDA_EXE="$(hash -r; which conda)"
    conda config --set solver libmamba

    # Install pytorch:
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    # Install other conda packages:
    # conda install -y rich -c conda-forge
    # Install other pip packages:
    pip install transformers datasets evaluate accelerate deepspeed rich simple-parsing
else
    conda activate $CONDA_ENV_PREFIX
fi


set -x  # print commands.

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:="configs/ds_level2.yaml"}
MODEL_NAME=${MODEL_NAME:="facebook/opt-2.7b"}
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:="1"}
OUTPUT_DIR=${OUTPUT_DIR:=$SCRATCH/logs/llm_training/$SLURM_JOB_ID}

mkdir -p $OUTPUT_DIR
conda env export > $OUTPUT_DIR/environment.yml

# Get a unique port for this job based on the job ID
export MASTER_PORT=${MASTER_PORT:=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))}
export MASTER_ADDR=${MASTER_ADDR:=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
export WORLD_SIZE=${WORLD_SIZE:=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))}
# TODO: Make sure this works correctly even with odd numbers of cpus / gpus / nodes (e.g. never zero).
CPUS_PER_GPU=${CPUS_PER_GPU:=$(($SLURM_CPUS_PER_TASK * SLURM_NTASKS / $WORLD_SIZE))}
export OMP_NUM_THREADS=$CPUS_PER_GPU

# mem_limit_in_bytes=$(cat /sys/fs/cgroup/memory/slurm/uid_"$(id -u)"/job_"${SLURM_JOBID}"/memory.limit_in_bytes)
# Enable storing the dataset in-memory.
# TODO: Turning this on actually seems to invalidate the cache dir, which sucks!
# export HF_DATASETS_IN_MEMORY_MAX_SIZE=$mem_limit_in_bytes


# NOTE: Uses `srun` to launch `accelerate launch` on each node with the right `--machine_rank`.
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --output=logs/slurm-%j_%t.out \
    bash -c 'accelerate launch \
    --config_file='$ACCELERATE_CONFIG' \
    --machine_rank=$SLURM_NODEID \
    --num_cpu_threads_per_process='$CPUS_PER_GPU' \
    --main_process_ip='$MASTER_ADDR' \
    --main_process_port='$MASTER_PORT' \
    --num_processes='$WORLD_SIZE' \
    main.py \
    --output_dir='$OUTPUT_DIR' \
    --config_name='$MODEL_NAME' --tokenizer_name='$MODEL_NAME' \
    --dataset_name=wikitext --dataset_config_name wikitext-103-v1 \
    --per_device_train_batch_size='$PER_GPU_BATCH_SIZE' --max_train_steps=1000 --with_tracking --report_to=wandb'
