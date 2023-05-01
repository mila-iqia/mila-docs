#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --signal=B:TERM@300 # tells the controller to send SIGTERM to the job 5
                            # min before its time ends to give it a chance for
                            # better cleanup. If you cancel the job manually,
                            # make sure that you specify the signal as TERM like
                            # so scancel --signal=TERM <jobid>.
                            # https://dhruveshp.com/blog/2021/signal-propagation-on-slurm/

# trap the signal to the main BATCH script here.
sig_handler()
{
	echo "BATCH interrupted"
	wait # wait for all children, this is important!
}

trap 'sig_handler' SIGINT SIGTERM SIGCONT


# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"


# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.7


# Creating the environment for the first time:
# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
#     pytorch-cuda=11.7 scipy -c pytorch -c nvidia
# Other conda packages:
# conda install -y -n pytorch -c conda-forge rich tqdm

# Activate pre-existing environment.
conda activate pytorch


# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
# General-purpose alternatives combining copy and unpack:
#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/


# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

# Execute Python script
python main.py
