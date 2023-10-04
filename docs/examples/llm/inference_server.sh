#!/bin/bash

#
# Assume you have conda installed
#
# Usage:
#   
#     sbatch --ntasks-per-node=1 --mem=32G inference_server_SHARED.sh meta/Llama-2-7b-chat-hf
#     sbatch --ntasks-per-node=2 --mem=64G inference_server_SHARED.sh meta/Llama-2-13b-chat-hf
#     sbatch --ntasks-per-node=8 --mem=192G inference_server_SHARED.sh meta/Llama-2-70b-chat-hf
#

#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G

usage() {
  echo "Usage: $0 [-m] [-p] 
  echo "  -h              Display this help message."
  echo "  -m MODEL        Specify a file to process."
  echo "  -p PATH         Specify a directory to work in."
  echo "  ARGUMENT        Any additional argument you want to process."
  exit 1
}

MODEL=""
PATH=""
ENV="./env"


while getopts ":hf:d:" opt; do
  case $opt in
    h)
      usage
      ;;
    m)
      MODEL="$OPTARG"
      ;;
    p)
      PATH="$OPTARG"
      ;;
    e)
      ENV="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done


export MILA_WEIGHTS="/network/weights/"

cd $SLURM_TMPDIR

#
#   Fix problem with conda saying it is not "init properly"
#
CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

#
#   Create a new environment
#
if [ ! -d "$ENV" ]; then
     conda create --prefix $ENV python=3.9 -y 
fi
conda activate $ENV
pip install vllm

PORT=$(python -c "import socket; sock = socket.socket(); sock.bind(('', 0)); print(sock.getsockname()[1])")
HOST="$(hostname)"
NAME="$WEIGHTS/$MODEL"

#
#   Save metadata for retrival
#
scontrol update job $SLURM_JOB_ID comment="model=$MODEL|host=$HOST|port=$PORT|shared=y"

# 
#   Launch Server
#
python -m vllm.entrypoints.openai.api_server       \
     --host $HOST                                  \
     --port $PORT                                  \
     --model "$MODEL"                              \
     --tensor-parallel-size $SLURM_NTASKS_PER_NODE \
     --served-model-name "$MODEL"
