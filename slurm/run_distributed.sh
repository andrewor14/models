#!/bin/bash

# =================================================
#  Example script for running models through slurm
# =================================================

# Set common configs
source common_configs.sh

export NUM_PARAMETER_SERVERS="0"
export NUM_WORKERS="4"
export NUM_GPUS_PER_WORKER="0"
export RESNET_SIZE="56"
export BATCH_SIZE="32"

./run_slurm.sh

