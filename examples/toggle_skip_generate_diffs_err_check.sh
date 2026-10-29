#!/bin/bash
# Use this to toggle the check of errors from docs/examples/perprocess.py
pushd `dirname "${BASH_SOURCE[0]}"` >/dev/null
_SCRIPT_DIR=`pwd -P`
popd >/dev/null

if [[ ! -f "${_SCRIPT_DIR}/generate_diffs.sh_err_ok" ]]
then
	echo "${_SCRIPT_DIR}/generate_diffs.sh_err_ok created"
	touch "${_SCRIPT_DIR}/generate_diffs.sh_err_ok"
else
	echo "${_SCRIPT_DIR}/generate_diffs.sh_err_ok removed"
	rm "${_SCRIPT_DIR}/generate_diffs.sh_err_ok"
fi
