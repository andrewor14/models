#!/bin/bash

source common.sh

export DATASET="${DATASET:=imagenet}"
if [[ "$DATASET" == "imagenet" ]]; then
  export RUN_FILE="resnet_imagenet_main.py"
  export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/imagenet}"
elif [[ "$DATASET" == "cifar10" ]]; then
  export RUN_FILE="resnet_cifar_main.py"
  export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/cifar10/cifar-10-batches-bin}"
else
  echo "ERROR: Unknown dataset '$DATASET'"
  exit 1
fi

set_job_name "resnet-$DATASET"
export CODE_DIR="${CODE_DIR:=$BASE_DIR/models/official/vision/image_classification}"
export TRAIN_DIR="${TRAIN_DIR:=${BASE_DIR}/train_data/${JOB_NAME}}"
export LOG_FILE="${LOG_FILE:=${LOG_DIR}/${JOB_NAME}.log}"

# Workload-specific flags
export NUM_GPUS="${NUM_GPUS:=1}"
export BATCH_SIZE="${BATCH_SIZE:=192}"
export NUM_EPOCHS="${NUM_EPOCHS:=90}"
export NUM_STEPS="${NUM_STEPS:=0}"
export EPOCHS_BETWEEN_EVALS="${EPOCHS_BETWEEN_EVALS:=4}"
export ENABLE_EAGER="${ENABLE_EAGER:=true}"
export ENABLE_CHECKPOINTS="${ENABLE_CHECKPOINTS:=false}"
export NUM_CHECKPOINTS_TO_KEEP="${NUM_CHECKPOINTS_TO_KEEP:=5}"
export SAVED_CHECKPOINT_DIR="${SAVED_CHECKPOINT_DIR:=}"

# Set distribution strategy
if [[ "$HOROVOD_ENABLED" == "true" ]]; then
  export DISTRIBUTION_STRATEGY="mirrored"
elif [[ "$NUM_NODES" > "1" ]]; then
  export DISTRIBUTION_STRATEGY="multi_worker_mirrored"
fi
export DISTRIBUTION_STRATEGY="${DISTRIBUTION_STRATEGY:=default}"

mkdir -p "$TRAIN_DIR"

print_diff_and_env > "$LOG_FILE" 2>&1

"$PYTHON_COMMAND" "${CODE_DIR}/${RUN_FILE}"\
  --num_gpus="$NUM_GPUS"\
  --data_dir="$DATA_DIR"\
  --batch_size="$BATCH_SIZE"\
  --train_steps="$NUM_STEPS"\
  --train_epochs="$NUM_EPOCHS"\
  --epochs_between_evals="$EPOCHS_BETWEEN_EVALS"\
  --skip_eval="$SKIP_EVAL"\
  --model_dir="$TRAIN_DIR"\
  --dtype="$DTYPE"\
  --enable_eager="$ENABLE_EAGER"\
  --enable_checkpoint_and_export="$ENABLE_CHECKPOINTS"\
  --num_checkpoints_to_keep="$NUM_CHECKPOINTS_TO_KEEP"\
  --saved_checkpoint_dir="$SAVED_CHECKPOINT_DIR"\
  --enable_xla="$ENABLE_XLA"\
  --log_steps="$LOG_STEPS"\
  --distribution_strategy="$DISTRIBUTION_STRATEGY"\
  --num_virtual_nodes_per_device="$NUM_VIRTUAL_NODES_PER_DEVICE"\
  >> "$LOG_FILE" 2>&1

