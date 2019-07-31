#!/bin/bash

# ================================================
#  Example script for running models in a cluster
# ================================================

# Set common configs
source common_configs.sh

# -----------------------------------------------------------------------------------
# Mode can be one of the following:
#   (1) static: number of workers are fixed throughout training
#   (2) checkpoint-restart: number of workers are dynamically adjusted by repeatedly
#     restoring from checkpoints, killing existing processes on restart
#   (3) autoscaling: number of workers are dynamically adjusted by spawning new
#     processes while keeping existing ones
# TODO: implement 'checkpoint-restart'
# -----------------------------------------------------------------------------------
export MODE="${MODE:=autoscaling}"
if [[ "$MODE" != "static" ]] &&\
    [[ "$MODE" != "checkpoint-restart" ]] &&\
    [[ "$MODE" != "autoscaling" ]]; then
  echo "ERROR: Unknown mode '$MODE'"
  exit 1
fi

# Models flags
export USE_KERAS="true"
export NUM_PARAMETER_SERVERS="0"
export NUM_WORKERS="${NUM_WORKERS:=4}"
export BATCH_SIZE="${BATCH_SIZE:=1024}"
export DATASET="${DATASET:=cifar10}"
if [[ "$USE_KERAS" == "true" ]]; then
  export SKIP_EVAL="true"
  export ENABLE_EAGER="true"
  export USE_HOROVOD="true"
  export LOG_STEPS="1"
else
  export RESNET_SIZE="56"
  export LOG_EVERY_N_STEPS="1"
fi
export RUN_TAG="${DATASET}-${BATCH_SIZE}-${MODE}"

# Autoscaling flags
if [[ "$MODE" == "autoscaling" ]]; then
  export AUTOSCALING_DISABLE_WRITE_GRAPH="true"
  export AUTOSCALING_DISABLE_CHECKPOINTS="true"
  export AUTOSCALING_DISABLE_CHECKPOINT_RESTORE="true"
  export AUTOSCALING_SPAWN_EVERY_N_STEPS="${AUTOSCALING_SPAWN_EVERY_N_STEPS:=10}"
  export AUTOSCALING_MAX_WORKERS="${AUTOSCALING_MAX_WORKERS:=60}"
  # If we're just trying to attach to an existing cluster, just launch 1 worker
  if [[ -n "$AUTOSCALING_MASTER_HOST_PORT" ]]; then
    export NUM_WORKERS=1
    export RUN_TAG="${RUN_TAG}-spawned"
  fi
fi

./deploy.sh

