#!/bin/bash

source common.sh

if [[ -z "$JOB_NAME" ]]; then
  set_job_name "bert-glue"
fi
maybe_set_spawn_log_file
export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/bert/finetuning}"
export BERT_MODEL_NAME="${BERT_MODEL_NAME:=uncased_L-12_H-768_A-12}"
export BERT_MODEL_DIR="${BERT_MODEL_DIR:=${BASE_DIR}/dataset/bert/${BERT_MODEL_NAME}}"
export BERT_CONFIG_FILE="${BERT_CONFIG_FILE:=${BERT_MODEL_DIR}/bert_config.json}"
export CHECKPOINT_NAME="${CHECKPOINT_NAME:=bert_model.ckpt}"
export SAVED_CHECKPOINT_PATH="${SAVED_CHECKPOINT_PATH:=${BERT_MODEL_DIR}/${CHECKPOINT_NAME}}"
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
  export EVAL_BATCH_SIZE="$BATCH_SIZE"
fi
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:=32}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:=32}"
export GLUE_TASK="${GLUE_TASK:=MRPC}"
export NUM_STEPS="${NUM_STEPS:=0}"
export NUM_EPOCHS="${NUM_EPOCHS:=$DEFAULT_NUM_EPOCHS}"
export STEPS_PER_LOOP="${STEPS_PER_LOOP:=1}"
export LEARNING_RATE="${LEARNING_RATE:=2e-5}"
export USE_KERAS_COMPILE_FIT="${USE_KERAS_COMPILE_FIT:=true}"
export DTYPE="fp32"

if [[ "$ENABLE_CHECKPOINTS" == "true" ]]; then
  export MODEL_EXPORT_PATH="${MODEL_EXPORT_PATH:=$TRAIN_DIR}"
fi

mkdir -p "$TRAIN_DIR"

print_diff_and_env > "$LOG_FILE" 2>&1

"$PYTHON_COMMAND" "${CODE_DIR}/run_classifier.py"\
  --mode="train_and_eval"\
  --input_meta_data_path="${DATA_DIR}/${GLUE_TASK}_meta_data"\
  --train_data_path="${DATA_DIR}/${GLUE_TASK}_train.tf_record"\
  --eval_data_path="${DATA_DIR}/${GLUE_TASK}_eval.tf_record"\
  --bert_config_file="$BERT_CONFIG_FILE"\
  --init_checkpoint="$SAVED_CHECKPOINT_PATH"\
  --model_dir="$TRAIN_DIR"\
  --train_batch_size="$TRAIN_BATCH_SIZE"\
  --eval_batch_size="$EVAL_BATCH_SIZE"\
  --num_train_steps="$NUM_STEPS"\
  --num_train_epochs="$NUM_EPOCHS"\
  --steps_per_loop="$STEPS_PER_LOOP"\
  --skip_eval="$SKIP_EVAL"\
  --learning_rate="$LEARNING_RATE"\
  --dtype="$DTYPE"\
  --enable_xla="$ENABLE_XLA"\
  --enable_checkpoints="$ENABLE_CHECKPOINTS"\
  --model_export_path="$MODEL_EXPORT_PATH"\
  --num_checkpoints_to_keep="$NUM_CHECKPOINTS_TO_KEEP"\
  --log_steps="$LOG_STEPS"\
  --distribution_strategy="$DISTRIBUTION_STRATEGY"\
  --use_keras_compile_fit="$USE_KERAS_COMPILE_FIT"\
  --num_virtual_nodes_per_device="$NUM_VIRTUAL_NODES_PER_DEVICE"\
  --enable_monitor_memory="$ENABLE_MONITOR_MEMORY"\
  --enable_elasticity="$ENABLE_ELASTICITY"\
  >> "$LOG_FILE" 2>&1

