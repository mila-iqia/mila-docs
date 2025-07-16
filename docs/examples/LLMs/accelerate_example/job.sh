#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16GB

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))

# Expect this variable to be set by the `safe_sbatch` submission script or similar.
# Otherwise, raises an error with a message.
GIT_COMMIT=${GIT_COMMIT:?"GIT_COMMIT must be set to the commit hash you want to run this job on. Use 'safe_sbatch' instead of 'sbatch' to submit this job."}

# This assumes that we're inside the project when we submit the job.
# Need to know where to go after cloning the repo in /tmp.
project_root=$(git rev-parse --show-toplevel)
project_dirname=$(basename $project_root)
submit_dir_relative_to_project=$(realpath --relative-to=$(dirname $project_root) $SLURM_SUBMIT_DIR)

# Use this on DRAC clusters where compute nodes don't have internet access.
# NOTE: make sure to run `uv sync` once before submitting the jobs.
# export UV_OFFLINE=1

srun --ntasks-per-node=1 bash -c "\
    git clone $project_root \$SLURM_TMPDIR/$project_dirname && \
    cd \$SLURM_TMPDIR/$project_dirname && \
    git checkout $GIT_COMMIT && \
    uv sync --directory=\$SLURM_TMPDIR/$submit_dir_relative_to_project"
# srun --nodes=1 --ntasks-per-node=1 uv sync --offline
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable
srun bash -c "\
    uv run --directory=\$SLURM_TMPDIR/$submit_dir_relative_to_project \
    accelerate launch \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines  $SLURM_NNODES --num_processes $SLURM_NTASKS \
    main.py $@"
