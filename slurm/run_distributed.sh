#!/bin/bash

# =================================================
#  Example script for running models through slurm
# =================================================

# Set common configs
source common_configs.sh

# If we're just trying to attach to an existing cluster, just launch 1 worker
if [[ -n "$AUTOSCALING_MASTER_HOST_PORT" ]]; then
  export NUM_WORKERS=1
  export RUN_TAG="distributed-added"
else
  export NUM_WORKERS=4
  export RUN_TAG="distributed"
fi

# Models flags
export NUM_PARAMETER_SERVERS="0"
export NUM_GPUS_PER_WORKER="0"
export RESNET_SIZE="56"
export BATCH_SIZE="32"

# Autoscaling flags
export AUTOSCALING_DISABLE_WRITE_GRAPH="true"
export AUTOSCALING_DISABLE_CHECKPOINT_RESTORE="true"

./run_slurm.sh

