#!/bin/bash
# Use this to update the diffs based on the contents of the files.

pushd `dirname "${BASH_SOURCE[0]}"` >/dev/null
_SCRIPT_DIR=`pwd -P`
popd >/dev/null

set -e

generate_diff() {
    echo "Generating diff for docs/examples/$1 -> docs/examples/$2"
    # NOTE: Assuming that this gets run from the `docs` folder (as is the case when building the docs).

    # Write a diff file to be shown in the documentation.

    echo " # $1 -> $2" > "$2.diff"
    git diff --no-index -U9999 \
        "$1" \
        "$2" \
        | grep -Ev "^--- |^\+\+\+ |^@@ |^index |^diff --git" \
        >> "$2.diff"
}

pushd "${_SCRIPT_DIR}" >/dev/null

# single_gpu -> multi_gpu
generate_diff distributed/single_gpu/job.sh distributed/multi_gpu/job.sh
generate_diff distributed/single_gpu/main.py distributed/multi_gpu/main.py

# multi_gpu -> multi_node
generate_diff distributed/multi_gpu/job.sh distributed/multi_node/job.sh
generate_diff distributed/multi_gpu/main.py distributed/multi_node/main.py

# single_gpu -> checkpointing
generate_diff distributed/single_gpu/job.sh good_practices/checkpointing/job.sh
generate_diff distributed/single_gpu/main.py good_practices/checkpointing/main.py

# single_gpu -> data
generate_diff distributed/single_gpu/job.sh good_practices/data/job.sh

# single_gpu -> hpo_with_orion
generate_diff distributed/single_gpu/job.sh good_practices/hpo_with_orion/job.sh
generate_diff distributed/single_gpu/main.py good_practices/hpo_with_orion/main.py

# single_gpu -> wandb_setup
generate_diff distributed/single_gpu/job.sh good_practices/wandb_setup/job.sh
generate_diff distributed/single_gpu/main.py good_practices/wandb_setup/main.py

popd >/dev/null
