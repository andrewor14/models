#!/bin/bash

export TIMESTAMP=`date +%m_%d_%y_%s%3N`
export JOB_NAME="resnet-imagenet-$TIMESTAMP"
export LOG_DIR="${LOG_DIR:=/home/andrew/Documents/dev/logs}"
export DATA_DIR="${DATA_DIR:=/home/andrew/Documents/dev/dataset/imagenet}"
export TRAIN_DIR="${TRAIN_DIR:=/home/andrew/Documents/dev/train_data/$JOB_NAME}"

export NUM_GPUS="${NUM_GPUS:=1}"
export BATCH_SIZE="${BATCH_SIZE:=192}"
export NUM_EPOCHS="${NUM_EPOCHS:=90}"
export NUM_STEPS="${NUM_STEPS:=0}"
export EPOCHS_BETWEEN_EVALS="${EPOCHS_BETWEEN_EVALS:=4}"
export SKIP_EVAL="${SKIP_EVAL:=false}"
export DTYPE="${DTYPE:=fp16}"
export ENABLE_EAGER="${ENABLE_EAGER:=true}"
export ENABLE_XLA="${ENABLE_XLA:=true}"

# Set this to 1 for memory allocation logs
export TF_CPP_MIN_VLOG_LEVEL=1

mkdir -p "$TRAIN_DIR"

python3 "/home/andrew/Documents/dev/models/official/vision/image_classification/resnet_imagenet_main.py"\
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
  --enable_xla="$ENABLE_XLA" > "$LOG_DIR/"$JOB_NAME".log" 2>&1

