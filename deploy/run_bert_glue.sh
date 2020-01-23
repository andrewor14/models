#!/bin/bash

source common.sh

export JOB_NAME="bert-glue-${TIMESTAMP}"
export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/bert/glue/finetuning_data}"
export TRAIN_DIR="${TRAIN_DIR:=${BASE_DIR}/train_data/${JOB_NAME}}"

if [[ -n "$BATCH_SIZE" ]]; then
  export TRAIN_BATCH_SIZE="$BATCH_SIZE"
  export EVAL_BATCH_SIZE="$BATCH_SIZE"
fi
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:=32}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:=32}"
export GLUE_TASK="${GLUE_TASK:=MRPC}"
export NUM_EPOCHS="${NUM_EPOCHS:=3}"
export NUM_STEPS="${NUM_STEPS:=0}"
export SKIP_EVAL="${SKIP_EVAL:=false}"
export STEPS_PER_LOOP="${STEPS_PER_LOOP:=1}"
export LEARNING_RATE="${LEARNING_RATE:=2e-5}"
export STRATEGY_TYPE="${STRATEGY_TYPE:=mirror}"
export DTYPE="${DTYPE:=fp16}"
export ENABLE_XLA="${ENABLE_XLA:=false}"

mkdir -p "$TRAIN_DIR"

python3 "${BERT_CODE_DIR}/run_classifier.py"\
  --mode="train_and_eval" \
  --input_meta_data_path="${DATA_DIR}/${GLUE_TASK}_meta_data" \
  --train_data_path="${DATA_DIR}/${GLUE_TASK}_train.tf_record" \
  --eval_data_path="${DATA_DIR}/${GLUE_TASK}_eval.tf_record" \
  --bert_config_file="${BERT_BASE_DIR}/bert_config.json" \
  --init_checkpoint="${BERT_BASE_DIR}/bert_model.ckpt" \
  --model_dir="$TRAIN_DIR" \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --eval_batch_size="$EVAL_BATCH_SIZE" \
  --num_train_epochs="$NUM_EPOCHS" \
  --num_train_steps="$NUM_STEPS" \
  --skip_eval="$SKIP_EVAL" \
  --steps_per_loop="$STEPS_PER_LOOP" \
  --learning_rate="$LEARNING_RATE" \
  --strategy_type="$STRATEGY_TYPE" \
  --dtype="$DTYPE"\
  --enable_xla="$ENABLE_XLA" > "${LOG_DIR}/${JOB_NAME}.log" 2>&1

