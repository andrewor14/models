#!/bin/bash

source common.sh

set_job_name "bert-pretraining"
export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/bert/pretrain}"
export INPUT_FILES="${INPUT_FILES:=${DATA_DIR}/tf_examples.tfrecord*}"
export BERT_CONFIG="${BERT_CONFIG_FILE:=${DATA_DIR}/bert_config.json}"
export CODE_DIR="${CODE_DIR:=$BASE_DIR/models/official/nlp/bert}"
export TRAIN_DIR="${TRAIN_DIR:=${BASE_DIR}/train_data/${JOB_NAME}}"
export LOG_FILE="${LOG_FILE:=${LOG_DIR}/${JOB_NAME}.log}"

# Workload-specific flags
if [[ -n "$BATCH_SIZE" ]]; then
  export TRAIN_BATCH_SIZE="$BATCH_SIZE"
fi
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:=32}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:=128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:=20}"
export NUM_STEPS_PER_EPOCH="${NUM_STEPS_PER_EPOCH:=1000}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:=3}"
export LEARNING_RATE:="${LEARNING_RATE:=2e-5}"

mkdir -p "$TRAIN_DIR"

print_diff_and_env > "$LOG_FILE" 2>&1

"$PYTHON_COMMAND" "${CODE_DIR}/run_pretraining.py"\
  --input_files="$INPUT_FILES"\
  --model_dir="$TRAIN_DIR"\
  --bert_config_file="$BERT_CONFIG_FILE"\
  --train_batch_size="$TRAIN_BATCH_SIZE"\
  --max_seq_length="$MAX_SEQ_LENGTH"\
  --max_predictions_per_seq="$MAX_PREDICTIONS_PER_SEQ"\
  --num_steps_per_epoch="$NUM_STEPS_PER_EPOCH"\
  --num_train_epochs="$NUM_TRAIN_EPOCHS"\
  --learning_rate="$LEARNING_RATE"\
  --num_virtual_nodes_per_device="$NUM_VIRTUAL_NODES_PER_DEVICE"\
  >> "$LOG_FILE" 2>&1

