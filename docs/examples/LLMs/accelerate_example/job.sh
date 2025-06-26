#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16GB

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# TODO: Make this work in a multi-node setting with code checkpointing
# (clone repo to $SLURM_TMPDIR and run things from there), without --nodes=1 so it runs on each node.
srun --nodes=1 --ntasks-per-node=1 uv sync --offline

# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable
srun uv run --offline bash -c "accelerate launch \
    --machine_rank \$SLURM_NODEID
    --multi_gpu \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines  $SLURM_NNODES --num_processes $SLURM_NTASKS \
    main.py $@"
