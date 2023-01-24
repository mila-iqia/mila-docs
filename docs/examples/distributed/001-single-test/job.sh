#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=unkillable

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module purge
module load anaconda/3

conda activate py38torch113

python main.py
