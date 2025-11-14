#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:59:00
#SBATCH --job-name=llm_training
#SBATCH --output=logs/slurm-%j.out

set -e  # exit on error.

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Make sure to use UV_OFFLINE=1 on DRAC clusters where compute nodes don't have internet access,
# or use `module load httpproxy` if it works.
# Note: You will either have to warm up the uv cache before submitting your job so  or use the drac wheelhouse as a source.
# export UV_OFFLINE=1

## Code checkpointing with git to avoid unexpected bugs ##
UV_DIR=$(./code_checkpointing.sh)
echo "Git commit used for this job: ${GIT_COMMIT:-not set - code checkpointing is not enabled}"
echo "Running uv commands in directory: $UV_DIR"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:="configs/ds_level2.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:=$SCRATCH/logs/llm_training/$SLURM_JOB_ID}

set -x  # print commands.


# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# NOTE: $SLURM_GPUS_ON_NODE is the number of GPUS on the *current* node, so this assumes that each
# node has the same # of allocated GPUS.
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE))

# TODO: Make sure this works correctly even with odd numbers of cpus / gpus / nodes (e.g. never zero).
export CPUS_PER_GPU=$(($SLURM_CPUS_PER_TASK * $SLURM_NTASKS / $WORLD_SIZE))
# NOTE: Setting this because `openmp` (called by `torch.distributed.run`, called by `accelerate launch`)
# otherwise sets it to 1, which might be bad for performance.
export OMP_NUM_THREADS=$CPUS_PER_GPU

NUM_NODES=${NUM_NODES:=$SLURM_JOB_NUM_NODES}

# Enable storing the dataset in-memory.
# mem_limit_in_bytes=$(cat /sys/fs/cgroup/memory/slurm/uid_"$(id -u)"/job_"${SLURM_JOBID}"/memory.limit_in_bytes)
# Note: Turning this on might increase performance, but invalidates the cache dir, which sucks!
# export HF_DATASETS_IN_MEMORY_MAX_SIZE=$mem_limit_in_bytes

# NOTE: When `--with_tracking` is passed, the `WANDB_API_KEY` environment variable must be set.
# You should ideally already have this in your ~/.bash_aliases file or similar.
export HF_HOME=${NUM_NODES:=$SCRATCH/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:=$SCRATCH/cache/huggingface/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:=$SCRATCH/cache/huggingface/hub}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# TODO: Probably best to only copy some files, not the entire cache.
# srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "\
#     mkdir -p \$SLURM_TMPDIR/cache && \
#     cp -r $HF_HOME \$SLURM_TMPDIR/cache/huggingface"

# unset HF_DATASETS_CACHE
# unset HUGGINGFACE_HUB_CACHE
# unset HF_HOME

# NOTE: Uses `srun` to launch `accelerate launch` on each node with the right `--machine_rank`.
srun --kill-on-bad-exit=1 --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --output=logs/slurm-%j_%t.out \
    bash -c "uv run --directory=$UV_DIR accelerate launch \
        --machine_rank=\$SLURM_NODEID \
        --config_file=$ACCELERATE_CONFIG \
        --num_cpu_threads_per_process=$CPUS_PER_GPU \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --num_processes=$WORLD_SIZE \
        --num_machines=$NUM_NODES \
        main.py \
        --output_dir=$OUTPUT_DIR \
        --with_tracking $@"

# Move any preprocessed dataset files back over to $SCRATCH so we don't have to recompute them every time.
rsync -a --progress $SLURM_TMPDIR/cache/huggingface/ $SCRATCH/cache/huggingface

