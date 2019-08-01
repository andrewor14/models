#!/bin/bash

# =============================================================
#  The main entry point to launching a tensorflow process.
#  This expects JOB_NAME to be set.
# =============================================================

# Set common configs
source common_configs.sh

if [[ -z "$JOB_NAME" ]]; then
  echo "ERROR: JOB_NAME must be set."
  exit 1
fi

if [[ -n "$NUM_GPUS" ]]; then
  echo "ERROR: Do not set NUM_GPUS. Set NUM_GPUS_PER_WORKER instead."
  exit 1
fi

# General configs
NUM_GPUS_PER_WORKER="${NUM_GPUS_PER_WORKER:=$DEFAULT_NUM_GPUS_PER_WORKER}"
NUM_EPOCHS="${NUM_EPOCHS:=100}"
BATCH_SIZE="${BATCH_SIZE:=32}"
EPOCHS_BETWEEN_EVALS="${EPOCHS_BETWEEN_EVALS:=10}"
DISTRIBUTION_STRATEGY="${DISTRIBUTION_STRATEGY:=multi_worker_mirrored}"
DATASET="${DATASET:=cifar10}"

# Dataset-specific configs
if [[ "$DATASET" == "cifar10" ]]; then
  DATA_DIR="$CIFAR10_DATA_DIR"
  if [[ "$USE_KERAS" == "true" ]]; then
    RUN_SCRIPT="$MODELS_DIR/official/resnet/keras/keras_cifar_main.py"
  else
    RUN_SCRIPT="$MODELS_DIR/official/resnet/cifar10_main.py"
  fi
elif [[ "$DATASET" == "imagenet" ]]; then
  DATA_DIR="$IMAGENET_DATA_DIR"
  RUN_SCRIPT="$MODELS_DIR/official/resnet/keras/keras_imagenet_main.py"
  if [[ "$USE_KERAS" != "true" ]]; then
    echo "ERROR: You must set USE_KERAS to 'true' for ImageNet training"
    exit 1
  fi
else
  echo "ERROR: Unknown dataset '$DATASET'"
  exit 1
fi

# Keras-specific configs
if [[ "$USE_KERAS" == "true" ]]; then
  SKIP_EVAL="${SKIP_EVAL:=false}"
  ENABLE_EAGER="${ENABLE_EAGER:=true}"
  USE_HOROVOD="${USE_HOROVOD:=false}"
  LOG_STEPS="${LOG_STEPS:=100}"
else
  RESNET_SIZE="${RESNET_SIZE:=56}"
  LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:=100}"
fi

# Set up working directories
TRAIN_DIR="${TRAIN_DIR:=$BASE_TRAIN_DIR/$JOB_NAME}"
mkdir -p "$TRAIN_DIR"

# If we're running horovod, then we're just using tensorflow's CollectiveAllReduceStrategy
# to update the variables, but we actually want to bypass the allreduce implementation in
# that strategy since we're already doing it in horovod
# TODO: This currently fails with workers with multiple GPUs. Fix it.
if [[ "$USE_HOROVOD" == "true" ]] && [[ "$NUM_GPUS_PER_WORKER" == "1" ]]; then
  export BYPASS_DISTRIBUTION_STRATEGY_ALLREDUCE="true"
fi

# Only allow positive number of parameter servers if we're running in parameter_server mode
if [[ "$DISTRIBUTION_STRATEGY" != "parameter_server" ]] && [[ "$NUM_PARAMETER_SERVERS" != "0" ]]; then
  echo "ERROR: NUM_PARAMETER_SERVERS must be 0 if we're not using 'parameter_server' distribution strategy"
  exit 1
fi

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

# Build flags
COMMON_FLAGS=""\
" --data_dir=$DATA_DIR"\
" --model_dir=$TRAIN_DIR"\
" --num_gpus=$NUM_GPUS_PER_WORKER"\
" --train_epochs=$NUM_EPOCHS"\
" --batch_size=$BATCH_SIZE"\
" --epochs_between_evals=$EPOCHS_BETWEEN_EVALS"\
" --distribution_strategy=$DISTRIBUTION_STRATEGY"

if [[ "$USE_KERAS" == "true" ]]; then
  FLAGS="$COMMON_FLAGS"\
" --skip_eval=$SKIP_EVAL"\
" --enable_eager=$ENABLE_EAGER"\
" --use_horovod=$USE_HOROVOD"\
" --log_steps=$LOG_STEPS"
else
  FLAGS="$COMMON_FLAGS"\
" --log_every_n_steps=$LOG_EVERY_N_STEPS"\
" --resnet_size=$RESNET_SIZE"
fi

# Run it
echo "Running $RUN_SCRIPT with the following flags: $FLAGS"
"$PYTHON_COMMAND" "$RUN_SCRIPT" $FLAGS

