#!/bin/bash
exec accelerate launch --multi_gpu                   \
                       --num_machines  $SLURM_NNODES \
                       --machine_rank  $SLURM_PROCID \
                       --num_processes $(( ${SLURM_NNODES:-1} * ${SLURM_GPUS_ON_NODE:-4} )) \
                       "$@"