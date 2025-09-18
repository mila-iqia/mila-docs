#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=200G  # We need 200GB of storage on the local disk of each node.
#SBATCH --time=05:00:00

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "Attempt #${SLURM_RESTART_COUNT:-0}"

# Make sure to use UV_OFFFLINE=1 on DRAC clusters where compute nodes don't have internet access,
# or use `module load httpproxy/1.0` if it works.
# Note: You will either have to warm up the uv cache before submitting your job so  or use the drac wheelhouse as a source.
export UV_OFFLINE=1
module load httpproxy/1.0

## Code checkpointing with git to avoid unexpected bugs ##
UV_DIR=$(./code_checkpointing.sh)
echo "Git commit used for this job: ${GIT_COMMIT:-not set - code checkpointing is not enabled}"
echo "Running uv commands in directory: $UV_DIR"

# Stage dataset into $SLURM_TMPDIR
# Prepare the dataset on each node's local storage using all the CPUs (and memory) of each node.
mkdir -p $SLURM_TMPDIR/data
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES:-1} bash -c \
    "uv run --directory=$UV_DIR python prepare_data.py --dest \$SLURM_TMPDIR/data"


# These environment variables are used by torch.distributed and should ideally be set
# before running the python script, or at the very beginning of the python script.
# Master address is the hostname of the first node in the job.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS

# srun is always used to launch the tasks.
# Whether there is one 'task' per GPU or one task per node can vary based on your setup.
# In the latter case, you would typically use torchrun or accelerate to launch one processes
# per GPU within each task.
# See the commented examples below for different ways to launch the training script.

# Important note: In all cases, some variables (for example RANK, LOCAL_RANK, or machine_rank
# in accelerate) vary between tasks, so we need to escape env variables such as $SLURM_PROCID,
# $SLURM_TMPDIR and $SLURM_NODEID so they are evaluated within each task, not just once here
# on the first node.

## Pure SLURM version ##
# They can either be set here or as early as possible in the Python script.
# Use `uv run --offline` on clusters without internet access on compute nodes.
# Using `srun` executes the command once per task, once per GPU in our case.
srun bash -c \
    "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    uv run --directory=$UV_DIR \
    python main.py --dataset_path=\$SLURM_TMPDIR/data $@"

## srun + torchrun version ##
## TODO: Use the right torchrun flags to get the equivalent of pure slurm / accelerate 
# srun --ntasks-per-node=1 bash -c "\
#     uv run --directory=$UV_DIR \
#     python -m torch.distributed.run \
#     --machine_rank \$SLURM_NODEID \
#     main.py $@"



## srun + accelerate version ##
## NOTE: This particular example doesn't use accelerate, this is just here to illustrate.
# srun --ntasks-per-node=1 bash -c "\
#     uv run --directory=$UV_DIR \
#     accelerate launch \
#     --machine_rank \$SLURM_NODEID \
#     --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
#     --num_machines  $SLURM_NNODES --num_processes $SLURM_NTASKS \
#     main.py $@"
