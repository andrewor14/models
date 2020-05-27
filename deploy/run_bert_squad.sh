#!/bin/bash

source common.sh

set_job_name "bert-squad"
export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/bert/finetuning}"
export PRETRAINED_DATA_DIR="${PRETRAINED_DATA_DIR:=${BASE_DIR}/dataset/bert/uncased_L-12_H-768_A-12}"
export CHECKPOINT_NAME="${CHECKPOINT_NAME:=bert_model.ckpt}"
export SAVED_CHECKPOINT_PATH="${SAVED_CHECKPOINT_PATH:=${PRETRAINED_DATA_DIR}/${CHECKPOINT_NAME}}"
export CODE_DIR="${CODE_DIR:=$BASE_DIR/models/official/nlp/bert}"
export TRAIN_DIR="${TRAIN_DIR:=${BASE_DIR}/train_data/${JOB_NAME}}"
export LOG_FILE="${LOG_FILE:=${LOG_DIR}/${JOB_NAME}.log}"

# Workload-specific flags
if [[ -n "$NUM_STEPS" ]]; then
  export DEFAULT_NUM_EPOCHS=1
else
  export DEFAULT_NUM_EPOCHS=3
fi
if [[ -n "$BATCH_SIZE" ]]; then
  export TRAIN_BATCH_SIZE="$BATCH_SIZE"
  export PREDICT_BATCH_SIZE="$BATCH_SIZE"
fi
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:=32}"
export PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:=32}"
export NUM_STEPS="${NUM_STEPS:=0}"
export NUM_EPOCHS="${NUM_EPOCHS:=$DEFAULT_NUM_EPOCHS}"
export STEPS_PER_LOOP="${STEPS_PER_LOOP:=1}"
export LEARNING_RATE="${LEARNING_RATE:=8e-5}"
export SQUAD_VERSION="${SQUAD_VERSION:=v1.1}"
export DTYPE="fp32"

if [[ "$ENABLE_CHECKPOINTS" == "true" ]]; then
  export MODEL_EXPORT_PATH="${MODEL_EXPORT_PATH:=$TRAIN_DIR}"
fi

mkdir -p "$TRAIN_DIR"

print_diff_and_env > "$LOG_FILE" 2>&1

"$PYTHON_COMMAND" "${CODE_DIR}/run_squad.py" \
  --input_meta_data_path="${DATA_DIR}/squad_${SQUAD_VERSION}_meta_data"\
  --train_data_path="${DATA_DIR}/squad_${SQUAD_VERSION}_train.tf_record"\
  --predict_file="${SQUAD_DIR}/dev-${SQUAD_VERSION}.json"\
  --vocab_file="${PRETRAINED_DATA_DIR}/vocab.txt"\
  --bert_config_file="${PRETRAINED_DATA_DIR}/bert_config.json"\
  --init_checkpoint="$SAVED_CHECKPOINT_PATH"\
  --model_dir="$TRAIN_DIR"\
  --train_batch_size="$TRAIN_BATCH_SIZE"\
  --predict_batch_size="$PREDICT_BATCH_SIZE"\
  --num_train_steps="$NUM_STEPS"\
  --num_train_epochs="$NUM_EPOCHS"\
  --steps_per_loop="$STEPS_PER_LOOP"\
  --learning_rate="$LEARNING_RATE"\
  --dtype="$DTYPE"\
  --enable_xla="$ENABLE_XLA"\
  --enable_checkpoints="$ENABLE_CHECKPOINTS"\
  --model_export_path="$MODEL_EXPORT_PATH"\
  --num_checkpoints_to_keep="$NUM_CHECKPOINTS_TO_KEEP"\
  --log_steps="$LOG_STEPS"\
  --distribution_strategy="$DISTRIBUTION_STRATEGY"\
  --num_virtual_nodes_per_device="$NUM_VIRTUAL_NODES_PER_DEVICE"\
  --enable_monitor_memory="$ENABLE_MONITOR_MEMORY"\
  >> "$LOG_FILE" 2>&1

