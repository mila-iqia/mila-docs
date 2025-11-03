#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Stage dataset into $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
srun --ntasks=${SLURM_JOB_NUM_NODES:-1} uv run python -c \
    'import os, torchvision.datasets; torchvision.datasets.CIFAR10(root=os.environ["SLURM_TMPDIR"] + "/data", download=True)'
# General-purpose alternatives combining copy and unpack:
#     unzip   /network/datasets/some/file.zip -d $SLURM_TMPDIR/data/
#     tar -xf /network/datasets/some/file.tar -C $SLURM_TMPDIR/data/

# =============
# Execute Orion
# =============

# Specify an experiment name with `-n`,
# which could be reused to display results (see section "Running example" below)

# Specify max trials (here 10) to prevent a too-long run.

# Then you can specify a search space for each `main.py`'s script parameter
# you want to optimize. Here we optimize only the learning rate.


# Configure Orion
# ===================
#
#    - use a pickleddb database stored in $SCRATCH
#    - worker dies if idle for more than a minute
#
export ORION_CONFIG=$SLURM_TMPDIR/orion-config.yml
cat > $ORION_CONFIG <<- EOM
    experiment:
        name: orion-example
        algorithms:
            tpe:
                seed: null
                n_initial_points: 5
        max_broken: 10
        max_trials: 10

    storage:
        database:
            host: $SCRATCH/orion.pkl
            type: pickleddb
EOM

srun --output=slurm-%A_%a_%t.out uv run orion hunt --config $ORION_CONFIG python main.py \
    --learning-rate~'loguniform(1e-5, 1.0)'
