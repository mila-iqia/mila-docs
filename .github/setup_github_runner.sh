#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/runner_%j.out

##
## GitHub Actions Self-Hosted Runner Setup Script
##
## This script downloads, configures, and starts a GitHub Actions self-hosted runner.
## It can be launched with `sbatch` on a Slurm cluster or run directly on a local machine.
##
## Prerequisites:
##   - SH_TOKEN environment variable must be set (GitHub token with admin permissions)
##   - Required commands: curl, tar, uvx (or python3), shasum
##   - For Slurm: SCRATCH and SLURM_TMPDIR environment variables must be set
##
## Environment Variables:
##   - SH_TOKEN: GitHub token with 'Administration' repository permission (required)
##   - SLURM_TMPDIR: Temporary directory on Slurm cluster (auto-set by Slurm)
##   - SCRATCH: Persistent storage on Slurm cluster (required if SLURM_TMPDIR is set)
##   - SLURM_CLUSTER_NAME: Cluster name for runner labels (optional)
##
# TODO: might cause issues if running this script on a local machine since $SCRATCH and
# $SLURM_TMPDIR won't be set.

set -o errexit
set -o nounset
set -o pipefail

# ============================================================================
# Configuration
# ============================================================================

readonly REPO="mila-iqia/mila-docs"
readonly RUNNER_VERSION="2.317.0"
readonly RUNNER_ARCHIVE="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
readonly RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_ARCHIVE}"
readonly EXPECTED_CHECKSUM="9e883d210df8c6028aff475475a457d380353f9d01877d51cc01a17b2a91161d"
readonly GITHUB_API_VERSION="2022-11-28"

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

log_error() {
    log "ERROR: $*" >&2
}

log_warning() {
    log "WARNING: $*" >&2
}

die() {
    log_error "$@"
    exit 1
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    log "Setting up environment..."

    # Source bash aliases if available (This is where the SH_TOKEN secret environment variable is set)
    if [ -f "$HOME/.bash_aliases" ]; then
        log "Sourcing $HOME/.bash_aliases"
        # shellcheck source=/dev/null
        source "$HOME/.bash_aliases"
    fi
}

# ============================================================================
# Validation Functions
# ============================================================================

check_required_commands() {
    log "Checking for required commands..."
    local missing_commands=()
    local required_commands=(curl python3 tar shasum)

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done

    if [ ${#missing_commands[@]} -gt 0 ]; then
        die "Missing required commands: ${missing_commands[*]}"
    fi

    log "All required commands are available"
}

check_github_token() {
    if [ -z "${SH_TOKEN:-}" ]; then
        die "SH_TOKEN environment variable is not set." \
            "This script requires a GitHub token with 'Administration' repository permission." \
            "See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens"
    fi
    log "GitHub token is set"
}

validate_slurm_environment() {
    if [ -n "${SLURM_TMPDIR:-}" ]; then
        if [ -z "${SCRATCH:-}" ]; then
            die "SCRATCH environment variable is not set but SLURM_TMPDIR is set." \
                "This script requires SCRATCH to be set when running on a Slurm cluster."
        fi
        log "Slurm environment detected: WORKDIR=$SLURM_TMPDIR, SCRATCH=$SCRATCH"
    fi
}

# ============================================================================
# Runner Download and Setup
# ============================================================================

determine_workdir() {
    if [ -n "${SLURM_TMPDIR:-}" ]; then
        echo "$SLURM_TMPDIR"
    else
        echo "$HOME/actions-runners/$REPO"
    fi
}

download_runner_archive() {
    local archive_path="$1"
    local archive_url="$2"

    if [ -f "$archive_path" ]; then
        log "Archive already exists: $archive_path"
        return 0
    fi

    log "Downloading runner archive from GitHub..."
    log "URL: $archive_url"
    log "Destination: $archive_path"

    if ! curl --fail --progress-bar --location --output "$archive_path" "$archive_url"; then
        die "Failed to download runner archive"
    fi

    log "Download completed successfully"
}

verify_archive_checksum() {
    local archive_path="$1"
    local expected_checksum="$2"

    log "Verifying archive checksum..."

    if ! echo "$expected_checksum  $archive_path" | shasum -a 256 -c >&2; then
        die "Archive checksum verification failed! The downloaded file may be corrupted."
    fi

    log "Checksum verification passed"
}

extract_runner_archive() {
    local archive_path="$1"
    local workdir="$2"

    # Check if already extracted
    if [ -d "$workdir/bin" ] && [ -f "$workdir/config.sh" ]; then
        log "Runner files already extracted, skipping extraction"
        return 0
    fi

    log "Extracting runner archive..."
    if ! tar -xzf "$archive_path" -C "$workdir"; then
        die "Failed to extract runner archive"
    fi

    log "Extraction completed successfully"
}

# ============================================================================
# GitHub API Functions
# ============================================================================

get_registration_token() {
    local repo="$1"
    local token="$2"

    log "Requesting registration token from GitHub API..."

    local temp_headers
    temp_headers=$(mktemp) || die "Failed to create temporary file"
    trap "rm -f '$temp_headers'" EXIT

    # Write API headers to temp file
    cat > "$temp_headers" <<EOF
Accept: application/vnd.github+json
Authorization: Bearer $token
X-GitHub-Api-Version: $GITHUB_API_VERSION
EOF
    
    # Make API request
    local api_url="https://api.github.com/repos/$repo/actions/runners/registration-token"
    local api_response
    api_response=$(curl --fail --silent --show-error --location \
        --request POST \
        --header "@$temp_headers" \
        "$api_url") || {
        rm -f "$temp_headers"
        die "Failed to request registration token from GitHub API"
    }

    # Extract token from JSON response
    local registration_token
    if ! registration_token=$(echo "$api_response" | python3 -c \
            "import sys, json; print(json.load(sys.stdin)['token'])" >&2) || \
       [ -z "$registration_token" ]; then
        rm -f "$temp_headers"
        die "Failed to parse GitHub API response. Response: $api_response"
    fi

    rm -f "$temp_headers"
    trap - EXIT

    log "Successfully obtained registration token (expires in 1 hour)"
    echo "$registration_token"
}

# ============================================================================
# Runner Configuration
# ============================================================================

build_runner_labels() {
    local cluster_name="${SLURM_CLUSTER_NAME:-}"

    if [ -n "$cluster_name" ]; then
        echo "$cluster_name-$SLURM_NNODES-${SLURM_GPUS_PER_NODE##*:},self-hosted"
    else
        echo "self-hosted"
    fi
}

configure_runner() {
    local workdir="$1"
    local repo="$2"
    local registration_token="$3"
    local labels="$4"
    local cluster_name="${SLURM_CLUSTER_NAME:-}"

    log "Configuring runner..."
    log "Repository: $repo"
    log "Labels: $labels"
    log "Cluster: ${cluster_name:-'(not set)'}"

    cd "$workdir" || die "Failed to change to workdir: $workdir"

    # Build config.sh command
    local config_cmd=(
        "./config.sh"
        --url "https://github.com/$repo"
        --token "$registration_token"
        --unattended
        --replace
        --labels "$labels"
    )

    # Add ephemeral flag if on Slurm cluster
    if [ -n "$cluster_name" ]; then
        config_cmd+=(--ephemeral)
        log "Configuring as ephemeral runner (Slurm cluster detected)"
    fi

    # Run configuration (don't fail if already configured)
    if ! "${config_cmd[@]}"; then
        log_warning "config.sh failed or runner already configured. Continuing anyway..."
    else
        log "Runner configured successfully"
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log "Starting GitHub Actions runner setup..."
    log "Repository: $REPO"
    log "Runner version: $RUNNER_VERSION"

    # Setup and validation
    setup_environment
    check_required_commands
    check_github_token
    validate_slurm_environment

    # Determine working directory
    local workdir
    workdir=$(determine_workdir)
    log "Working directory: $workdir"
    mkdir -p "$workdir" || die "Failed to create workdir: $workdir"

    # Download and setup runner
    local archive_path
    if [ -n "${SLURM_TMPDIR:-}" ]; then
        # On Slurm: store archive in SCRATCH, symlink to SLURM_TMPDIR
        archive_path="$SCRATCH/$RUNNER_ARCHIVE"
        download_runner_archive "$archive_path" "$RUNNER_URL"

        # Create symlink in workdir if needed
        if [ ! -L "$workdir/$RUNNER_ARCHIVE" ]; then
            ln -s "$archive_path" "$workdir/$RUNNER_ARCHIVE" || \
                die "Failed to create symlink"
        fi
        archive_path="$workdir/$RUNNER_ARCHIVE"
    else
        # Local: store archive in workdir
        archive_path="$workdir/$RUNNER_ARCHIVE"
        download_runner_archive "$archive_path" "$RUNNER_URL"
    fi

    # Verify and extract
    verify_archive_checksum "$archive_path" "$EXPECTED_CHECKSUM"
    extract_runner_archive "$archive_path" "$workdir"

    # Get registration token
    local registration_token
    registration_token=$(get_registration_token "$REPO" "$SH_TOKEN")

    # Configure runner
    local labels
    labels=$(build_runner_labels)
    configure_runner "$workdir" "$REPO" "$registration_token" "$labels"

    # Set environment variables for GitHub Actions. These are normally be set by
    # GitHub Actions when running in the cloud, so they are visible in the
    # python script. Unclear why GitHub Actions doesn't set these on a
    # self-hosted runner.
    export GITHUB_ACTIONS="true"
    export RUNNER_LABELS="$labels"

    log "Starting GitHub Actions runner..."
    log "Runner labels: $RUNNER_LABELS"
    log "GITHUB_ACTIONS: $GITHUB_ACTIONS"

    # Change to workdir and start runner
    cd "$workdir" || die "Failed to change to workdir: $workdir"

    # Launch the runner (exec replaces current process)
    log "Launching runner process..."
    exec ./run.sh
}

# Run main function
main "$@"

