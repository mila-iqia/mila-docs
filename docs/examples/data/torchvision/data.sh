#!/bin/bash
set -o errexit

function ln_files {
	# Clone the dataset structure of `src` to `dest` with symlinks and using
	# `workers` numbre of workers (defaults to 4)
	local src=$1
	local dest=$2
	local workers=${3:-4}

	(cd "${src}" && find -L * -type f) | while read f
	do
		mkdir --parents "${dest}/$(dirname "$f")"
		# echo source first so it is matched to the ln's '-T' argument
		readlink --canonicalize "${src}/$f"
		# echo output last so ln understands it's the output file
		echo "${dest}/$f"
	done | xargs -n2 -P${workers} ln --symbolic --force -T
}

_SRC=$1
_WORKERS=$2
# Referencing $SLURM_TMPDIR here instead of job.sh makes sure that the
# environment variable will only be resolved on the worker node (i.e. not
# referencing the $SLURM_TMPDIR of the master node)
_DEST=$SLURM_TMPDIR/data

ln_files "${_SRC}" "${_DEST}" ${_WORKERS}

# Reorganise the files if needed
(
	cd "${_DEST}"
	# Torchvision expects these names
	mv train.tar.gz 2021_train.tgz
	mv val.tar.gz 2021_valid.tgz
)

# Extract and prepare the data
python3 data.py "${_DEST}"
