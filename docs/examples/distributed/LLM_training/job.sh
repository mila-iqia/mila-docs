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

module --quiet purge

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:="configs/ds_level2.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:=$SCRATCH/logs/llm_training/$SLURM_JOB_ID}

# 'setup_env.sh' should be called before launching the job to create the
# environment and install packages only once in an environment where internet is
# accessible
source setup_env.sh

set -x  # print commands.


# Get a unique port for this job based on the job ID
export MASTER_PORT=${MASTER_PORT:=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))}
export MASTER_ADDR=${MASTER_ADDR:=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
# NOTE: $SLURM_GPUS_ON_NODE is the number of GPUS on the *current* node, so this assumes that each
# node has the same # of allocated GPUS.
export WORLD_SIZE=${WORLD_SIZE:=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))}

# TODO: Make sure this works correctly even with odd numbers of cpus / gpus / nodes (e.g. never zero).
export CPUS_PER_GPU=${CPUS_PER_GPU:=$(($SLURM_CPUS_PER_TASK * $SLURM_NTASKS / $WORLD_SIZE))}
# NOTE: Setting this because `openmp` (called by `torch.distributed.run`, called by `accelerate launch`)
# otherwise sets it to 1, which might be bad for performance.
export OMP_NUM_THREADS=$CPUS_PER_GPU

NUM_NODES=${NUM_NODES:=$SLURM_JOB_NUM_NODES}

# Enable storing the dataset in-memory.
# mem_limit_in_bytes=$(cat /sys/fs/cgroup/memory/slurm/uid_"$(id -u)"/job_"${SLURM_JOBID}"/memory.limit_in_bytes)
# Note: Turning this on might increase performance, but invalidates the cache dir, which sucks!
# export HF_DATASETS_IN_MEMORY_MAX_SIZE=$mem_limit_in_bytes

# TODO: When `--with_tracking` is passed, the `WANDB_API_KEY` environment variable must be set.

export HF_HOME=$SCRATCH/cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub

srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "\
    mkdir -p \$SLURM_TMPDIR/cache && \
    cp -r $HF_HOME \$SLURM_TMPDIR/cache/huggingface"

# unset HF_DATASETS_CACHE
# unset HUGGINGFACE_HUB_CACHE
# unset HF_HOME

# NOTE: Uses `srun` to launch `accelerate launch` on each node with the right `--machine_rank`.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1


cmd=(
    accelerate launch
    --machine_rank=\$SLURM_NODEID
    --config_file=$ACCELERATE_CONFIG
    --num_cpu_threads_per_process=$CPUS_PER_GPU
    --main_process_ip=$MASTER_ADDR
    --main_process_port=$MASTER_PORT
    --num_processes=$WORLD_SIZE
    --num_machines=$NUM_NODES
    main.py
    --output_dir=$OUTPUT_DIR
    --with_tracking "$@"
)
srun --kill-on-bad-exit=1 --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --output=logs/slurm-%j_%t.out \
    bash -c "$(for a in "${cmd[@]}" ; do echo -n \"$a\" "" ; done)"

# Move any preprocessed dataset files back over to $SCRATCH so we don't have to recompute them every time.
rsync -a --progress $SLURM_TMPDIR/cache/huggingface/ $SCRATCH/cache/huggingface
# srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --output=logs/slurm-%j_%t.out \
#    bash -c accelerate launch \
#     --machine_rank=\$SLURM_NODEID \
#     --config_file=$ACCELERATE_CONFIG \
#     --num_cpu_threads_per_process=$CPUS_PER_GPU \
#     --main_process_ip=$MASTER_ADDR \
#     --main_process_port=$MASTER_PORT \
#     --num_processes=$WORLD_SIZE \
#     --num_machines=$NUM_NODES \
#     main.py \
#     --output_dir=$OUTPUT_DIR \
#     --max_train_steps=100 --with_tracking "$@"
