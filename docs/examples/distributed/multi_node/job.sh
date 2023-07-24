#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.7

# Activate pre-existing environment.
# NOTE: Use the `make_env.sh` script to create the environment if you haven't already.
conda activate pytorch

# Stage dataset into $SLURM_TMPDIR (only on the first worker of each node)
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c \
   'mkdir -p $SLURM_TMPDIR/data && ln -s /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/'

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

# Execute Python script in each task (one per GPU)
srun python main.py
