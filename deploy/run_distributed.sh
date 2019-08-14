#!/bin/bash

# ============================================
#  A script for running models in the cluster
# ============================================

# Set common configs
source common_configs.sh

# -----------------------------------------------------------------------------------
# Mode can be one of the following:
#   (1) static: number of workers are fixed throughout training
#   (2) checkpoint-restart: number of workers are dynamically adjusted by repeatedly
#     restoring from checkpoints, killing existing processes on restart
#   (3) autoscaling: number of workers are dynamically adjusted by spawning new
#     processes while keeping existing ones
# -----------------------------------------------------------------------------------
export MODE="${MODE:=autoscaling}"
if [[ "$MODE" != "static" ]] &&\
    [[ "$MODE" != "checkpoint-restart" ]] &&\
    [[ "$MODE" != "autoscaling" ]]; then
  echo "ERROR: Unknown mode '$MODE'"
  exit 1
fi

# Models flags
export USE_KERAS="${USE_KERAS:=true}"
export NUM_PARAMETER_SERVERS="${NUM_PARAMETER_SERVERS:=0}"
export NUM_WORKERS="${NUM_WORKERS:=4}"
export DATASET="${DATASET:=cifar10}"
export BATCH_SIZE="${BATCH_SIZE:=1024}"

# Keras-specific flags
if [[ "$USE_KERAS" == "true" ]]; then
  export RUN_EAGERLY="${RUN_EAGERLY:=false}"
  export USE_HOROVOD="${USE_HOROVOD:=true}"
  export LOG_STEPS="1"
else
  export RESNET_SIZE="${RESNET_SIZE:=56}"
  export LOG_EVERY_N_STEPS="1"
fi

# Autoscaling flags
if [[ "$MODE" == "autoscaling" ]] || [[ "$MODE" == "checkpoint-restart" ]]; then
  if [[ "$MODE" == "autoscaling" ]]; then
    export AUTOSCALING_DISABLE_WRITE_GRAPH="true"
    export AUTOSCALING_DISABLE_CHECKPOINTS="true"
    export AUTOSCALING_DISABLE_CHECKPOINT_RESTORE="true"
    # If we're just trying to attach to an existing cluster, just launch 1 worker
    if [[ -n "$AUTOSCALING_MASTER_HOST_PORT" ]]; then
      export NUM_WORKERS=1
    fi
  fi
  export AUTOSCALING_SPAWN_EVERY_N_STEPS="${AUTOSCALING_SPAWN_EVERY_N_STEPS:=10}"
  export AUTOSCALING_MAX_WORKERS="${AUTOSCALING_MAX_WORKERS:=60}"
fi

# Set a unique job name to identify the experiment
function set_job_name() {
  export SUBMIT_TIMESTAMP="${SUBMIT_TIMESTAMP:=$(get_submit_timestamp)}"
  export RUN_TAG="${DATASET}-${BATCH_SIZE}-${MODE}-${NUM_WORKERS}"
  if [[ -n "$AUTOSCALING_MASTER_HOST_PORT" ]]; then
    export RUN_TAG="${RUN_TAG}-spawned"
  fi
  export JOB_NAME="${RUN_TAG}-${SUBMIT_TIMESTAMP}"
}

# In 'checkpoint-restart' mode, the job exits when adjusting its number of workers
# Therefore, we have to run tensorflow in a loop and exit only when training is done
if [[ "$MODE" == "checkpoint-restart" ]]; then
  while true; do
    set_job_name
    export TRAIN_DIR="$BASE_TRAIN_DIR/$JOB_NAME"
    ./deploy.sh
    # If the checkpoint metadata file does not exist, then assume that training is done
    # TODO: do we need to copy these checkpoint files to all the workers?
    METADATA_FILE="$TRAIN_DIR/checkpoint.metadata"
    if [[ ! -f "$METADATA_FILE" ]]; then
      break
    fi
    # Export all environment variables from the metadata file for the next run
    while read LINE; do
      ENV_VAR_NAME="$(echo $LINE | awk '{print $1}')"
      ENV_VAR_VALUE="$(echo $LINE | awk '{print $2}')"
      export "$ENV_VAR_NAME"="$ENV_VAR_VALUE"
    done < "$METADATA_FILE"
    export AUTOSCALING_CHECKPOINT_DIR="$TRAIN_DIR"
  done
else
  set_job_name
  ./deploy.sh
fi

