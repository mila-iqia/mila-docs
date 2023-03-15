#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -e  # exit on error.

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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# Execute Python script in each task (one per GPU)
# srun --tasks-per-node=1 bash -c 'accelerate launch --config_file='$ACCELERATE_CONFIG' --machine_rank=$SLURM_NODEID --main_process_ip='$SLURMD_NODENAME' example_train_script.py'
accelerate launch --config_file=docs/examples/distributed/LLM_training/a100_cluster_config.yaml \
    docs/examples/distributed/LLM_training/deepspeed_with_config_support.py \
    --config_name=facebook/opt-13b --tokenizer_name=facebook/opt-13b \
    --dataset_name=wikitext --dataset_config_name wikitext-103-v1 \
    --per_device_train_batch_size=1 --max_train_steps=100 --output_dir=output

