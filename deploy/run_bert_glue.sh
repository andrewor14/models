#!/bin/bash

source common.sh

set_job_name "bert-glue"
export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/bert/glue/finetuning_data}"
export PRETRAINED_DATA_DIR="${PRETRAINED_DATA_DIR:=$BASE_DIR/dataset/bert/uncased_L-12_H-768_A-12}"
export CHECKPOINT_NAME="$(ls "$PRETRAINED_DATA_DIR" | grep index | sed 's/\.index//g')"
export CODE_DIR="${CODE_DIR:=$BASE_DIR/models/official/nlp/bert}"
export TRAIN_DIR="${TRAIN_DIR:=${BASE_DIR}/train_data/${JOB_NAME}}"
export LOG_FILE="${LOG_FILE:=${LOG_DIR}/${JOB_NAME}.log}"

# Workload-specific flags
if [[ -n "$BATCH_SIZE" ]]; then
  export TRAIN_BATCH_SIZE="$BATCH_SIZE"
  export EVAL_BATCH_SIZE="$BATCH_SIZE"
fi
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:=32}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:=32}"
export GLUE_TASK="${GLUE_TASK:=MRPC}"
export NUM_EPOCHS="${NUM_EPOCHS:=3}"
export NUM_STEPS="${NUM_STEPS:=0}"
export STEPS_PER_LOOP="${STEPS_PER_LOOP:=1}"
export LEARNING_RATE="${LEARNING_RATE:=2e-5}"
export STRATEGY_TYPE="${STRATEGY_TYPE:=mirror}"
export USE_KERAS_COMPILE_FIT="${USE_KERAS_COMPILE_FIT:=true}"

mkdir -p "$TRAIN_DIR"

print_diff_and_env > "$LOG_FILE" 2>&1

"$PYTHON_COMMAND" "${CODE_DIR}/run_classifier.py"\
  --mode="train_and_eval"\
  --input_meta_data_path="${DATA_DIR}/${GLUE_TASK}_meta_data"\
  --train_data_path="${DATA_DIR}/${GLUE_TASK}_train.tf_record"\
  --eval_data_path="${DATA_DIR}/${GLUE_TASK}_eval.tf_record"\
  --bert_config_file="${PRETRAINED_DATA_DIR}/bert_config.json"\
  --init_checkpoint="${PRETRAINED_DATA_DIR}/${CHECKPOINT_NAME}"\
  --model_dir="$TRAIN_DIR"\
  --train_batch_size="$TRAIN_BATCH_SIZE"\
  --eval_batch_size="$EVAL_BATCH_SIZE"\
  --num_train_epochs="$NUM_EPOCHS"\
  --num_train_steps="$NUM_STEPS"\
  --skip_eval="$SKIP_EVAL"\
  --steps_per_loop="$STEPS_PER_LOOP"\
  --learning_rate="$LEARNING_RATE"\
  --strategy_type="$STRATEGY_TYPE"\
  --dtype="$DTYPE"\
  --enable_xla="$ENABLE_XLA"\
  --use_keras_compile_fit="$USE_KERAS_COMPILE_FIT"\
  --num_virtual_nodes_per_device="$NUM_VIRTUAL_NODES_PER_DEVICE"\
  >> "$LOG_FILE" 2>&1

