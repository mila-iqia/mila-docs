#!/bin/bash
exec accelerate launch --machine_rank $SLURM_NODEID "$@"
