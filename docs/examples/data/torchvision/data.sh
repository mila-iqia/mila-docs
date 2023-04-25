#!/bin/bash
set -o errexit

_SRC=$1
_DEST=$2
_WORKERS=$3

# Clone the dataset structure locally and reorganise the raw files if needed
(cd "${_SRC}" && find -L * -type f) | while read f
do
	mkdir --parents "${_DEST}/$(dirname "$f")"
	# echo source first so it is matched to the ln's '-T' argument
	readlink --canonicalize "${_SRC}/$f"
	# echo output last so ln understands it's the output file
	echo "${_DEST}/$f"
done | xargs -n2 -P${_WORKERS} ln --symbolic --force -T

(
	cd "${_DEST}"
	# Torchvision expects these names
	mv train.tar.gz 2021_train.tgz
	mv val.tar.gz 2021_valid.tgz
)

# Extract and prepare the data
python3 data.py "${_DEST}"
