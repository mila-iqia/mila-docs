#!/bin/bash

#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G

function usage() {
  echo "Usage: $0 [-m] [-p]"
  echo "  -h              Display this help message."
  echo "  -m MODEL        Specify the model name"
  echo "  -p PATH         Specify the model weights"
  echo "  -e ENV          Specify the conda environementt to use."
  echo "  ARGUMENT        Any additional argument you want to process."
  exit 1
}

MODEL=""
MODEL_PATH=""
ENV="./env"


while getopts ":hm:p:e:" opt; do
  case $opt in
    h)
      usage
      ;;
    m)
      MODEL="$OPTARG"
      ;;
    p)
      MODEL_PATH="$OPTARG"
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

echo "model: $MODEL"
echo " path: $MODEL_PATH"
echo "  env: $ENV"

export MILA_WEIGHTS="/network/weights/"
cd $SLURM_TMPDIR

#
#   Fix problem with conda saying it is not "init properly"
#
CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
CONDA_ENVS="$CONDA_BASE/../envs"
source $CONDA_BASE/../etc/profile.d/conda.sh

#
#   Create a new environment
#
if [ ! -d "$ENV" ] && [ "$ENV" != "base" ] && [ ! -d "$CONDA_ENVS/$ENV" ]; then
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
     --model "$MODEL_PATH"                         \
     --tensor-parallel-size $SLURM_NTASKS_PER_NODE \
     --served-model-name "$MODEL"
