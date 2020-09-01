#!/bin/bash

source common.sh

if [[ -z "$JOB_NAME" ]]; then
  set_job_name "bert-pretraining"
fi
maybe_set_spawn_log_file
export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/bert/pretraining}"
export BERT_MODEL_NAME="${BERT_MODEL_NAME:=uncased_L-12_H-768_A-12}"
export BERT_MODEL_DIR="${BERT_MODEL_DIR:=${BASE_DIR}/dataset/bert/${BERT_MODEL_NAME}}"
export BERT_CONFIG_FILE="${BERT_CONFIG_FILE:=${BERT_MODEL_DIR}/bert_config.json}"
export INPUT_FILES="${INPUT_FILES:=${DATA_DIR}/tf_examples.tfrecord*}"
export CODE_DIR="${CODE_DIR:=$BASE_DIR/models/official/nlp/bert}"
export TRAIN_DIR="${TRAIN_DIR:=${BASE_DIR}/train_data/${JOB_NAME}}"
export LOG_FILE="${LOG_FILE:=${LOG_DIR}/${JOB_NAME}.log}"

if [[ "$ENABLE_ELASTICITY" == "true" ]]; then
  echo "Elasticity is not supported for BERT pre-training"
  exit 1
fi

# Workload-specific flags
if [[ -n "$NUM_STEPS" ]]; then
  export DEFAULT_NUM_EPOCHS=1
else
  export DEFAULT_NUM_EPOCHS=3
fi
if [[ -n "$BATCH_SIZE" ]]; then
  export TRAIN_BATCH_SIZE="$BATCH_SIZE"
fi
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:=32}"
export NUM_STEPS="${NUM_STEPS:=1000}"
export NUM_EPOCHS="${NUM_EPOCHS:=$DEFAULT_NUM_EPOCHS}"
export STEPS_PER_LOOP="${STEPS_PER_LOOP:=1}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:=128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:=20}"
export LEARNING_RATE="${LEARNING_RATE:=2e-5}"
export DTYPE="fp32"

# The custom training loop exports its own checkpoints
export ENABLE_CHECKPOINTS="false"

mkdir -p "$TRAIN_DIR"

print_diff_and_env > "$LOG_FILE" 2>&1

"$PYTHON_COMMAND" "${CODE_DIR}/run_pretraining.py"\
  --input_files="$INPUT_FILES"\
  --bert_config_file="$BERT_CONFIG_FILE"\
  --model_dir="$TRAIN_DIR"\
  --train_batch_size="$TRAIN_BATCH_SIZE"\
  --num_train_steps="$NUM_STEPS"\
  --num_train_epochs="$NUM_EPOCHS"\
  --steps_per_loop="$STEPS_PER_LOOP"\
  --learning_rate="$LEARNING_RATE"\
  --dtype="$DTYPE"\
  --enable_xla="$ENABLE_XLA"\
  --enable_checkpoints="$ENABLE_CHECKPOINTS"\
  --log_steps="$LOG_STEPS"\
  --distribution_strategy="$DISTRIBUTION_STRATEGY"\
  --num_virtual_nodes_per_device="$NUM_VIRTUAL_NODES_PER_DEVICE"\
  --enable_monitor_memory="$ENABLE_MONITOR_MEMORY"\
  --max_seq_length="$MAX_SEQ_LENGTH"\
  --max_predictions_per_seq="$MAX_PREDICTIONS_PER_SEQ"\
  >> "$LOG_FILE" 2>&1

