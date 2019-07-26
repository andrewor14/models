#!/bin/bash

# ================================================
#  Example script for running models in a cluster
# ================================================

# Set common configs
source common_configs.sh

# Models flags
export USE_KERAS="true"
export NUM_PARAMETER_SERVERS="0"
export NUM_GPUS_PER_WORKER="0"
export BATCH_SIZE="1024"
if [[ "$USE_KERAS" == "true" ]]; then
  export SKIP_EVAL="true"
  export ENABLE_EAGER="true"
  export USE_HOROVOD="true"
  export LOG_STEPS="1"
else
  export RESNET_SIZE="56"
  export LOG_EVERY_N_STEPS="1"
fi

# Autoscaling flags
export AUTOSCALING_DISABLE_WRITE_GRAPH="true"
export AUTOSCALING_DISABLE_CHECKPOINTS="true"
export AUTOSCALING_DISABLE_CHECKPOINT_RESTORE="true"
export AUTOSCALING_SPAWN_EVERY_N_STEPS=10
export AUTOSCALING_MAX_WORKERS=60

# Set run tag
if [[ "$USE_HOROVOD" == "true" ]]; then
  export RUN_TAG="horovod"
else
  export RUN_TAG="distributed"
fi

# If we're just trying to attach to an existing cluster, just launch 1 worker
if [[ -n "$AUTOSCALING_MASTER_HOST_PORT" ]]; then
  export NUM_WORKERS=1
  export RUN_TAG="$RUN_TAG-added"
else
  export NUM_WORKERS="${NUM_WORKERS:=4}"
fi

./deploy.sh

