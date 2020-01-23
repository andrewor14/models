#!/bin/bash

source common.sh

export JOB_NAME="resnet-imagenet-${TIMESTAMP}"
export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/imagenet}"
export TRAIN_DIR="${TRAIN_DIR:=${BASE_DIR}/train_data/${JOB_NAME}}"

export NUM_GPUS="${NUM_GPUS:=1}"
export BATCH_SIZE="${BATCH_SIZE:=192}"
export NUM_EPOCHS="${NUM_EPOCHS:=90}"
export NUM_STEPS="${NUM_STEPS:=0}"
export EPOCHS_BETWEEN_EVALS="${EPOCHS_BETWEEN_EVALS:=4}"
export SKIP_EVAL="${SKIP_EVAL:=false}"
export DTYPE="${DTYPE:=fp16}"
export ENABLE_EAGER="${ENABLE_EAGER:=true}"
export ENABLE_XLA="${ENABLE_XLA:=false}"

mkdir -p "$TRAIN_DIR"

python3 "${RESNET_CODE_DIR}/resnet_imagenet_main.py"\
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
  --enable_xla="$ENABLE_XLA" > "${LOG_DIR}/${JOB_NAME}.log" 2>&1

