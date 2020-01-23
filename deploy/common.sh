#!/bin/bash

export TIMESTAMP=`date +%m_%d_%y_%s%3N`
export BASE_DIR="/home/andrew/Documents/dev"
export LOG_DIR="${LOG_DIR:=$BASE_DIR/logs}"
export RESNET_CODE_DIR="${RESNET_CODE_DIR:=$BASE_DIR/models/official/vision/image_classification}"
export BERT_BASE_DIR="${BERT_BASE_DIR:=$BASE_DIR/dataset/bert/uncased_L-12_H-768_A-12}"
export BERT_CODE_DIR="${BERT_CODE_DIR:=$BASE_DIR/models/official/nlp/bert}"

export LOG_MEMORY_ENABLED="${LOG_MEMORY_ENABLED:=true}"
if [[ "$LOG_MEMORY_ENABLED" == "true" ]]; then
  # Set this to 1 for memory allocation logs
  export TF_CPP_MIN_VLOG_LEVEL="1"
fi

