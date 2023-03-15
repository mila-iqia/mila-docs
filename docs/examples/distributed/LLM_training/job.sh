#!/bin/bash
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --time=01:00:00

set -e  # exit on error.
set -x  # print commands.


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3

if [ ! -d "$SLURM_TMPDIR/env" ]; then
    # Create a conda environment and use the libmamba solver:
    conda create -y -p $SLURM_TMPDIR/env python=3.9 conda conda-libmamba-solver -c conda-forge
    conda activate $SLURM_TMPDIR/env
    export CONDA_EXE="$(hash -r; which conda)"
    conda config --set solver libmamba

    # Install the pytorch dependencies:
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

    # Other conda packages:
    # conda install -y rich -c conda-forge
    # Other pip packages:
    pip install rich transformers datasets evaluate accelerate deepspeed
else
    conda activate $SLURM_TMPDIR/env
fi

module load cuda/11.7

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:=4}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# export CPATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/include/

# ACCELERATE_CONFIG="gabriele_config.yaml"
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:="a100_cluster_config.yaml"}

# TODO: This should be run once per node with `srun --ntasks-per-node=1 bash -c '...'`
accelerate launch \
    --config_file=$ACCELERATE_CONFIG \
    --machine_rank=$SLURM_NODEID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=$SLURM_NTASKS \
    deepspeed_with_config_support.py \
    --config_name=facebook/opt-13b --tokenizer_name=facebook/opt-13b \
    --dataset_name=wikitext --dataset_config_name wikitext-103-v1 \
    --per_device_train_batch_size=1 --max_train_steps=10 --output_dir=output

