#!/bin/bash

# =============================================================
#  The main entry point to launching a tensorflow process
#
#  This expects the following environment variables to be set:
#  JOB_NAME and NUM_WORKERS
# =============================================================

# Set common configs
source common_configs.sh

if [[ -z "$JOB_NAME" ]]; then
  echo "ERROR: JOB_NAME must be set."
  exit 1
fi

if [[ -z "$NUM_WORKERS" ]]; then
  echo "ERROR: NUM_WORKERS must be set."
  exit 1
fi

if [[ -n "$NUM_GPUS" ]]; then
  echo "ERROR: Do not set NUM_GPUS. Set NUM_GPUS_PER_WORKER instead."
  exit 1
fi

# General configs
NUM_GPUS_PER_WORKER="${NUM_GPUS_PER_WORKER:=$DEFAULT_NUM_GPUS_PER_WORKER}"
NUM_EPOCHS="${NUM_EPOCHS:=100}"
RESNET_SIZE="${RESNET_SIZE:=56}"
BATCH_SIZE="${BATCH_SIZE:=32}"
EPOCHS_BETWEEN_EVALS="${EPOCHS_BETWEEN_EVALS:=10}"
DISTRIBUTION_STRATEGY="${DISTRIBUTION_STRATEGY:=multi_worker_mirrored}"

# Only allow positive number of parameter servers if we're running in parameter_server mode
if [[ "$DISTRIBUTION_STRATEGY" != "parameter_server" ]] && [[ "$NUM_PARAMETER_SERVERS" != "0" ]]; then
  echo "ERROR: NUM_PARAMETER_SERVERS must be 0 if we're not using 'parameter_server' distribution strategy"
  exit 1
fi

# Set up working directories
TRAIN_DIR="${TRAIN_DIR:=$BASE_TRAIN_DIR/$JOB_NAME}"
mkdir -p "$TRAIN_DIR"

# Print diff and environment variables
DIFF="$(git diff)"
if [[ -n "$DIFF" ]]; then
  echo -e "\n=========================================================================="
  echo -e "git diff"
  echo -e "--------------------------------------------------------------------------"
  echo -e "$DIFF"
  echo -e "==========================================================================\n"
fi
echo -e "\n=========================================================================="
echo -e "My environment variables:"
echo -e "--------------------------------------------------------------------------"
printenv
echo -e "==========================================================================\n"

if [[ "$SINGLE_PROCESS_MODE" == "true" ]]; then
  unset SLURM_JOB_NODELIST
fi

"$PYTHON_COMMAND" $MODELS_DIR/official/resnet/cifar10_main.py\
  --data_dir="$CIFAR10_DATA_DIR"\
  --model_dir="$TRAIN_DIR"\
  --num_gpus="$NUM_GPUS_PER_WORKER"\
  --train_epochs="$NUM_EPOCHS"\
  --resnet_size="$RESNET_SIZE"\
  --batch_size="$BATCH_SIZE"\
  --epochs_between_evals="$EPOCHS_BETWEEN_EVALS"\
  --distribution_strategy="$DISTRIBUTION_STRATEGY"

