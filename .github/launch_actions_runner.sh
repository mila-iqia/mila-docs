#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/runner_%j.out

## This script can be used to launch a new self-hosted GitHub runner.
## It can be launched with `sbatch` on a SLURM cluster, or run directly on a local machine.
## It assumes that the SH_TOKEN environment variable can be read from $HOME/.bash_aliases
## and contains a GitHub token that is used to authenticate with the GitHub API to launch
## a new self-hosted runner.
set -euo pipefail
set -o errexit
# todo: might cause issues if running this script on a local machine since $SCRATCH and
# $SLURM_TMPDIR won't be set.
set -o nounset

# Seems to be required for the `uvx` to be found. (adds $HOME/.cargo/bin to PATH)
source $HOME/.cargo/env
# This is where the SH_TOKEN secret environment variable is set.
source $HOME/.bash_aliases

readonly repo="mila-iqia/mila-docs"
readonly action_runner_version="2.317.0"
readonly expected_checksum_for_version="9e883d210df8c6028aff475475a457d380353f9d01877d51cc01a17b2a91161d"


# Check for required commands.
for cmd in curl tar uvx; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is not installed."
        exit 1
    fi
done

if [ -z "${SH_TOKEN:-}" ]; then
    echo "Error: SH_TOKEN environment variable is not set."
    echo "This script requires the SH_TOKEN environment variable be set to a GitHub token with permissions to create new self-hosted runners for the current repository."
    echo "To create this token, Follow the docs here: "
    echo " - https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token"
    echo " - and click here to create the new token: https://github.com/settings/personal-access-tokens/new"
    echo "The fine-grained token must have the 'Administration - repository permissions (write)' scope."
    exit 1
fi

archive="actions-runner-linux-x64-$action_runner_version.tar.gz"

# Look for the actions-runner archive.
# 1. If SLURM_TMPDIR is set:
#     - set WORKDIR to $SLURM_TMPDIR
#     - check if the archive doesn't exist in $SCRATCH
#     - if it doesn't exist, download the archive from GitHub.
#     - Make a symlink to it in $SLURM_TMPDIR.
# 2. Otherwise, use ~/actions-runners/$repo as the WORKDIR, and download the archive from GitHub if
#it isn't already there.

if [ -n "${SLURM_TMPDIR:-}" ]; then
    # This was launched with sbatch on a SLURM cluster.
    WORKDIR=$SLURM_TMPDIR
    if [ ! -f "$SCRATCH/$archive" ]; then
        curl --fail -o "$SCRATCH/$archive" \
            -L "https://github.com/actions/runner/releases/download/v$action_runner_version/$archive"
    fi
    if [ ! -L "$WORKDIR/$archive" ]; then
        ln -s "$SCRATCH/$archive" "$WORKDIR/$archive"
    fi
else
    # This was launched as a script on a local or dev machine or in a non-SLURM environment.
    WORKDIR=$HOME/actions-runners/$repo
    mkdir -p $WORKDIR
    if [ ! -f "$WORKDIR/$archive" ]; then
        curl --fail -o "$WORKDIR/$archive" \
            -L "https://github.com/actions/runner/releases/download/v$action_runner_version/$archive"
    fi
fi
echo "Setting up self-hosted runner in $WORKDIR"
cd $WORKDIR


# Check the archive integrity.
echo "$expected_checksum_for_version  $archive" | shasum -a 256 -c
# Extract the installer
tar xzf $archive
# Use the GitHub API to get a temporary registration token for a new self-hosted runner.
# This requires you to be an admin of the repository and to have the $SH_TOKEN secret set to a
# github token with (ideally only) the appropriate permissions.
# https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-a-repository
# Example output:
# {
#   "token": "XXXXX",
#   "expires_at": "2020-01-22T12:13:35.123-08:00"
# }
t=$(tempfile) || exit
trap "rm -f -- '$t'" EXIT

# Write headers to the tempfile
cat <<EOF > "$t"
Accept: application/vnd.github+json
Authorization: Bearer $SH_TOKEN
X-GitHub-Api-Version: 2022-11-28
EOF

# Uses `uvx python` to just get python. Assumes that `uv` is already installed.
TOKEN=`curl --fail -L \
  -X POST \
  -H @$t \
  https://api.github.com/repos/$repo/actions/runners/registration-token | \
  uvx python -c "import sys, json; print(json.load(sys.stdin)['token'])"`

rm -f -- "$t"
trap - EXIT


# IF SLURM_CLUSTER_NAME is set, we're on a SLURM cluster, so configure the worker with --ephemeral.
export cluster=${SLURM_CLUSTER_NAME:-}
echo "Cluster name: $cluster"
# Create the runner and configure it programmatically with the token we just got
# from the GitHub API.

# For now, don't exit if the runner is already configured.
# This way, we might have more than one github runner job running at once.
./config.sh --url https://github.com/$repo --token $TOKEN \
  --unattended --replace --labels $cluster self-hosted ${SLURM_CLUSTER_NAME:+--ephemeral} || true

# Setting these environment variables which are normally be set by gh-actions when running in the cloud,
# so they are visible in the python script. Unclear why gh-actions doesn't set these on a self-hosted runner.
export GITHUB_ACTIONS="true"
export RUNNER_LABELS="self-hosted,$cluster"

# Launch the actions runner.
exec ./run.sh
