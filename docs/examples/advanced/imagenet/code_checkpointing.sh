#!/bin/bash
## Code checkpointing utility script ##
# When used in conjunction with `safe_sbatch` to submit jobs, this prevents
# changes in the python files between the job submission and job start from causing 
# unexpected bugs. This also greatly helps reproducibility of your experiments.

# Usage:
# - This should be called from within a SLURM sbatch job script.
# - This clones the project on each node's local storage at the current commit.
# - This creates the virtual environment for the project at that commit on each node's local storage using UV.
# - This returns the directory that should then be passed to the --directory argument of `uv run`
#   commands in the rest of the job script.

# Assumptions:
# - This assumes that we're inside the project when submitting a job.
# - This assumes that the project uses Git and UV (https://docs.astral.sh/uv).

set -e  # exit on error.

# We need to know where to go after cloning the repo in /tmp.
project_root=$(git rev-parse --show-toplevel)
project_dirname=$(basename $project_root)
submit_dir_relative_to_parent=$(realpath --relative-to=$(dirname $project_root) ${SLURM_SUBMIT_DIR:-$(pwd)})

# Expect this GIT_COMMIT variable to be set by the `safe_sbatch` submission script or similar.

# The directory where UV commands should be executed.
# - If code checkpointing is not used, this is the current directory.
# - If code checkpointing is used, this is the path from the parent folder of the project root
#   to the current directory where the job is submitted. The same relative path is recreated
#   with $SLURM_TMPDIR as a base.
UV_DIR="."
if [[ -n "$GIT_COMMIT" ]]; then
    # GIT_COMMIT is set, so we clone the repo in $SLURM_TMPDIR at that commit.
    echo "Job will run with code from commit $GIT_COMMIT" >&2
    UV_DIR="\$SLURM_TMPDIR/$submit_dir_relative_to_parent"
    srun --ntasks-per-node=1 bash -c "\
        git clone $project_root \$SLURM_TMPDIR/$project_dirname && \
        cd \$SLURM_TMPDIR/$project_dirname && \
        git checkout --detach $GIT_COMMIT && \
        uv sync --directory=$UV_DIR"
elif [[ -n "$(git -C $project_root status --porcelain)" ]]; then
    echo "Warning: GIT_COMMIT is not set and the current repo at $project_root has uncommitted changes." >&2
    echo "This may cause future jobs to fail or produce inconsistent results!" >&2
    echo "Consider using the 'safe_sbatch' script to submit jobs instead." >&2
else
    echo "GIT_COMMIT environment variable is not set, but the repo state is clean. " >&2
    echo "If you modify the files in the repo, future jobs might fail or produce inconsistent results. " >&2
fi
# return the UV_DIR variable as an output of this script.
echo $UV_DIR 
