#!/bin/bash
set -eof pipefail
git_status=`git status --porcelain`
# idea: Could add command-line arguments to control whether to add all changes and commit before sbatch.
if [[ ! -z $git_status ]]; then
    echo "Your working directory is dirty! Please add and commit changes before continuing."
    exit 1
fi;
# This environment variable will be available in the job script.
# It should be used to checkout the repo at this commit (in a different directory than the original).
# For example:
# ```
# git clone "$repo" "$dest"
# echo "Checking out commit $GIT_COMMIT"
# cd "$dest"
# git checkout $GIT_COMMIT
# ```
export GIT_COMMIT=`git rev-parse HEAD`
exec sbatch "$@"
