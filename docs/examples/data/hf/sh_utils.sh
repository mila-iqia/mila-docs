#!/bin/bash
set -o errexit

function _list_dataset_files {
	local _src=$1
	local _globs=("${@:2}")

	if [[ -z "${_globs[@]}" ]]
	then
		_globs=("*")
	else
		# Safely wrap globs in '"'
		_globs=("${_globs[@]/#/\"}")
		_globs=("${_globs[@]/%/\"}")
		_globs=("${_globs[@]//\*/\"\*\"}")
	fi

	pushd "${_src}" >/dev/null
	bash -c "find -L ${_globs[*]} -type f"
	popd >/dev/null
}

function _pair_files_src_to_dest {
	local _src=$1
	local _dest=$2

	while read _f
	do
		mkdir --parents "${_dest}/$(dirname "$_f")"
		# echo source first so it is matched to the cp's '-T' argument
		readlink --canonicalize "${_src}/$_f"
		# echo output last so cp understands it's the output file
		echo "${_dest}/$_f"
	done <&0
}

function _cp {
	local _src=$1
	local _dest=$2
	[[ -L "${_dest}" ]] && rm "${_dest}"
	cp --update --force -T "${_src}" "${_dest}"
}

function cp_files {
	local _src=$1
	local _dest=$2
	local _workers=$3
	local _globs=("${@:4}")

	while read _glob
	do
		_globs+=("$_glob")
	done <&0

	(export -f _cp
	 _list_dataset_files "${_src}" "${_globs[@]}" | \
	 	_pair_files_src_to_dest "${_src}" "${_dest}" | \
	 	xargs -n2 -P${_workers} bash -c '_cp "$@"' _)
}

function ln_files {
	local _src=$1
	local _dest=$2
	local _workers=$3
	local _globs=("${@:4}")

	_list_dataset_files "${_src}" "${_globs[@]}" | \
		_pair_files_src_to_dest "${_src}" "${_dest}" | \
		xargs -n2 -P${_workers} ln --symbolic --force -T
}

if [[ ! -z "$@" ]]
then
	"$@"
fi
