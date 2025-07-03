#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16GB

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))

# https://stackoverflow.com/a/1371283/6388696
project_dirname=${PWD##*/}


# TODO: Make this work in a multi-node setting with code checkpointing
# (clone repo to $SLURM_TMPDIR and run things from there), without --nodes=1 so it runs on each node.


# Expect this variable to be set, or raise an error with a message
GIT_COMMIT=${GIT_COMMIT:?"GIT_COMMIT must be set to the commit hash you want to run this job on. Use 'safe_sbatch' instead of 'sbatch' to submit this job."}

srun --ntasks-per-node=1 bash -c "\
    git clone \$SLURM_SUBMIT_DIR \$SLURM_TMPDIR/$project_dirname && \
    cd \$SLURM_TMPDIR/$project_dirname && \
    git checkout $GIT_COMMIT && \
    uv sync --offline"
# srun --nodes=1 --ntasks-per-node=1 uv sync --offline
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable
srun bash -c "\
    uv run --directory \$SLURM_TMPDIR/$project_dirname --offline accelerate launch \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines  $SLURM_NNODES --num_processes $SLURM_NTASKS \
    main.py $@"
