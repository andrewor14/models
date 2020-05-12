#!/bin/bash

source common.sh

export DATA_DIR="${DATA_DIR:=${BASE_DIR}/dataset/bert/glue}"
export OUTPUT_DIR="${OUTPUT_DIR:=${BASE_DIR}/dataset/bert/glue/finetuning_data}"
export TASK_NAME="${TASK_NAME:=MRPC}"

python3 "${BERT_CODE_DIR}/create_finetuning_data.py" \
 --input_data_dir="${DATA_DIR}/${TASK_NAME}" \
 --vocab_file="${BERT_BASE_DIR}/vocab.txt" \
 --train_data_output_path="${OUTPUT_DIR}/${TASK_NAME}_train.tf_record" \
 --eval_data_output_path="${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record" \
 --meta_data_file_path="${OUTPUT_DIR}/${TASK_NAME}_meta_data" \
 --fine_tuning_task_type=classification \
 --max_seq_length=128 \
 --classification_task_name="$TASK_NAME"

