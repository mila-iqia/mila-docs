#!/bin/bash
set -o errexit

_SRC=$1
_DEST=$2
_WORKERS=$3

# Clone the dataset structure (not the data itself) locally so HF can find the
# cache hashes it looks for. Else HF might think he needs to redo some
# preprocessing. Directories will be created and symlinks will replace the files
bash sh_utils.sh ln_files "${_SRC}" "${_DEST}" $_WORKERS

# Copy the preprocessed dataset to compute node's local dataset cache dir so it
# is close to the GPUs for faster training. Since HF can very easily change the
# hash to reference a preprocessed dataset, we only copy the data for the
# current preprocess pipeline.
python3 get_dataset_cache_files.py | bash sh_utils.sh cp_files "${_SRC}" "${_DEST}" $_WORKERS
