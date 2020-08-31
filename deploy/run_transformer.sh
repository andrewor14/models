#!/bin/bash

source common.sh

if [[ -z "$JOB_NAME" ]]; then
  set_job_name "transformer"
fi
maybe_set_spawn_log_file
export PARAM_SET="${PARAM_SET:=big}"
export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/transformer}"
export TRAIN_DIR="${TRAIN_DIR:=${BASE_DIR}/train_data/${JOB_NAME}}"
export VOCAB_FILE="${VOCAB_FILE:=${DATA_DIR}/vocab.ende.32768}"
export BLEU_SOURCE="${BLEU_SOURCE:=${DATA_DIR}/newstest2014.en}"
export BLEU_REF="${BLEU_REF:=${DATA_DIR}/newstest2014.de}"
export CODE_DIR="${CODE_DIR:=${BASE_DIR}/models/official/nlp/transformer}"
export LOG_FILE="${LOG_FILE:=${LOG_DIR}/${JOB_NAME}.log}"

# Workload-specific flags
export NUM_STEPS="${NUM_STEPS:=100000}"
export BATCH_SIZE="${BATCH_SIZE:=4096}"
export DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:=32}"
export MAX_LENGTH="${MAX_LENGTH:=64}"
export STEPS_BETWEEN_EVALS="${STEPS_BETWEEN_EVALS:=$NUM_STEPS}"

if [[ "$SKIP_EVAL" == "true" ]]; then
  export BLEU_SOURCE=""
  export BLEU_REF=""
fi

mkdir -p "$TRAIN_DIR"

print_diff_and_env > "$LOG_FILE" 2>&1

"$PYTHON_COMMAND" "${CODE_DIR}/transformer_main.py"\
  --num_gpus="$NUM_GPUS"\
  --param_set="$PARAM_SET"\
  --data_dir="$DATA_DIR"\
  --model_dir="$TRAIN_DIR"\
  --vocab_file="$VOCAB_FILE"\
  --batch_size="$BATCH_SIZE"\
  --decode_batch_size="$DECODE_BATCH_SIZE"\
  --train_steps="$NUM_STEPS"\
  --steps_between_evals="$STEPS_BETWEEN_EVALS"\
  --max_length="$MAX_LENGTH"\
  --bleu_source="$BLEU_SOURCE"\
  --bleu_ref="$BLEU_REF"\
  --distribution_strategy="$DISTRIBUTION_STRATEGY"\
  --enable_time_history="true"\
  --log_steps="1"\
  --num_virtual_nodes_per_device="$NUM_VIRTUAL_NODES_PER_DEVICE"\
  --enable_checkpointing="$ENABLE_CHECKPOINTS"\
  --num_checkpoints_to_keep="$NUM_CHECKPOINTS_TO_KEEP"\
  --enable_monitor_memory="$ENABLE_MONITOR_MEMORY"\
  --enable_elasticity="$ENABLE_ELASTICITY"\
  >> "$LOG_FILE" 2>&1

