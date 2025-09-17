#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=32G
#SBATCH --tmp=200G  # We need 200GB of storage on the local disk of each node.
#SBATCH --time=05:00:00
#SBATCH --output=/network/scratch/n/normandf/checkpoints/%j/out.txt

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "Attempt #${SLURM_RESTART_COUNT:-0}"


# Stage dataset into $SLURM_TMPDIR
# Prepare the dataset on each node's local storage using all the CPUs (and memory) of each node.
mkdir -p $SLURM_TMPDIR/data
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES:-1} bash -c "uv run \
    python prepare_data.py --dest \$SLURM_TMPDIR/data"

# These environment variables are used by torch.distributed and should ideally be set
# before running the python script, or at the very beginning of the python script.

# Master address is the hostname of the first node in the job.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS
# The RANK and LOCAL_RANK variables vary between tasks.
# Add these variables so that torch.distributed works out of the box.
# They can either be set here or as early as possible in the Python script.
# Use `uv run --offline` on clusters without internet access on compute nodes.
# Using `srun` executes the command once per task, once per GPU in our case.
srun bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID uv run \
    python main.py --dataset_path=\$SLURM_TMPDIR/data $@"
