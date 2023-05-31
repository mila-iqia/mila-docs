#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --tmp=1500G
set -o errexit

function wrap_cmd {
	for a in "$@"
	do
		echo -n "\"$a\" "
	done
}


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
# conda install -y -n pytorch -c conda-forge rich tqdm datasets

# Activate pre-existing environment.
conda activate pytorch


if [[ -z "$HF_DATASETS_CACHE" ]]
then
	# Store the huggingface datasets cache in $SCRATCH
	export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
fi
if [[ -z "$HUGGINGFACE_HUB_CACHE" ]]
then
	# Store the huggingface hub cache in $SCRATCH
	export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub
fi
if [[ -z "$_DATA_PREP_WORKERS" ]]
then
	_DATA_PREP_WORKERS=$SLURM_JOB_CPUS_PER_NODE
fi
if [[ -z "$_DATA_PREP_WORKERS" ]]
then
	_DATA_PREP_WORKERS=16
fi


# Preprocess the dataset and cache the result such that the heavy work is done
# only once *ever*
# Required conda packages:
# conda install -y -c conda-forge zstandard
srun --ntasks=1 --ntasks-per-node=1 \
	time -p python3 prepare_data.py "/network/datasets/pile" $_DATA_PREP_WORKERS


# Copy the preprocessed dataset to $SLURM_TMPDIR so it is close to the GPUs for
# faster training. This should be done once per compute node
cmd=(
	# Having 'bash' here allows the execution of a script file which might not
	# have the execution flag on
	bash data.sh
	# The current dataset cache dir
	"$HF_DATASETS_CACHE"
	# The local dataset cache dir
	# Use '' to lazy expand the expression such as $SLURM_TMPDIR will be
	# interpreted on the local compute node rather than the master node
	'$SLURM_TMPDIR/data'
	$_DATA_PREP_WORKERS
)
# 'time' will objectively give a measure for the copy of the dataset. This can
# be used to compare the timing of multiple code versionw and make sure any slow
# down doesn't come from the code itself.
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
	time -p bash -c "$(wrap_cmd "${cmd[@]}")"


# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

# Execute Python script
env_var=(
	# Use the local copy of the preprocessed dataset
	HF_DATASETS_CACHE='"$SLURM_TMPDIR/data"'
)
cmd=(
	python3
	main.py
)
srun bash -c "$(echo "${env_var[@]}") $(wrap_cmd "${cmd[@]}")"
