#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3

# Creating the environment for the first time:
# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
#     pytorch-cuda=11.6 -c pytorch -c nvidia
# Other conda packages:
# conda install -y -n pytorch -c conda-forge rich

# Activate pre-existing environment.
conda activate pytorch

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Execute Python script in each task (one per GPU)
srun --tasks-per-node=1 bash -c 'accelerate launch --config_file='$ACCELERATE_CONFIG' --machine_rank=$SLURM_NODEID --main_process_ip='$SLURMD_NODENAME' example_train_script.py'
