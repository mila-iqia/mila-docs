#!/bin/bash
set -v

# Usage:
#
#   sbatch --array=1-100 --gres=gpu:1 --cpus-per-gpu=2 --mem-per-gpu=16G hpo_launcher.sh train.py
#

# Environment
# ===================

export EXPERIMENT_NAME="MyExperiment"

# Constant
export DATASET_DEST=$SLURM_TMPDIR/dataset
export CHECKPOINT_PATH=$SCRATCH/checkpoint
export ORION_CONFIG=$SLURM_TMPDIR/orion-config.yml
export SPACE_CONFIG=$SCRATCH/space-config.json

# Configure Orion
# ===================
#
#    - use hyperband
#    - launch one worker per CPU allocated to the GPU
#    - worker dies if idle for more than a minute
#    - all workers share a single GPU to maximize usage
#
cat > $ORION_CONFIG <<- EOM
    experiment:
        name: ${EXPERIMENT_NAME}
        algorithms:
            hyperband:
                seed: null
        max_broken: 10

    worker:
        n_workers: $SLURM_CPUS_PER_GPU
        pool_size: 0
        executor: joblib
        heartbeat: 120
        max_broken: 10
        idle_timeout: 60

    storage:
        database:
            host: $SCRATCH/${EXPERIMENT_NAME}_orion.pkl
            type: pickleddb
EOM

# Define the hyperparameter search space
cat > $SPACE_CONFIG <<- EOM
    {
        "epochs": "orion~fidelity(1, 100, base=2)",
        "lr": "orion~loguniform(1e-5, 1.0)",
        "weight_decay": "orion~loguniform(1e-10, 1e-3)",
        "momentum": "orion~loguniform(0.9, 1.0)"
    }
EOM

# Run
# ===================

uv run orion hunt --config $ORION_CONFIG python "$@" --config $SPACE_CONFIG
