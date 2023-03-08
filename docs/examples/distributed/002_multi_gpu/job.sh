#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module purge
module load anaconda/3


# Activate pre-existing environment.
conda activate py38torch113


# Stage dataset into $SLURM_TMPDIR
cp -a /network/datasets/cifar10.var/cifar10_torchvision $SLURM_TMPDIR

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Execute Python script in distributed fashion
srun python main.py
