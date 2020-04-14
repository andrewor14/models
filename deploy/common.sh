#!/bin/bash

export TIMESTAMP=`date +%m_%d_%y_%s%3N`
export BASE_DIR="/home/andrew/Documents/dev"
export TF_DIR="$BASE_DIR/tensorflow"
export LOG_DIR="${LOG_DIR:=$BASE_DIR/logs}"
export RESNET_CODE_DIR="${RESNET_CODE_DIR:=$BASE_DIR/models/official/vision/image_classification}"
export BERT_BASE_DIR="${BERT_BASE_DIR:=$BASE_DIR/dataset/bert/uncased_L-12_H-768_A-12}"
export BERT_CODE_DIR="${BERT_CODE_DIR:=$BASE_DIR/models/official/nlp/bert}"
export TF_DEALLOCATE_OP_PATH="${TF_DEALLOCATE_OP_PATH:=$TF_DIR/bazel-bin/tensorflow/core/user_ops/deallocate.so}"

export LOG_MEMORY_ENABLED="${LOG_MEMORY_ENABLED:=false}"
if [[ "$LOG_MEMORY_ENABLED" == "true" ]]; then
  # Set this to 1 for memory allocation logs
  export TF_CPP_MIN_VLOG_LEVEL="1"
fi

set_job_name() {
  JOB_NAME="$1"
  if [[ -n "$RUN_TAG" ]]; then
    JOB_NAME="${JOB_NAME}-${RUN_TAG}"
  fi
  export JOB_NAME="${JOB_NAME}-${TIMESTAMP}"
}

print_diff() {
  echo -e "My commit is $(git log --oneline | head -n 1) ($PWD)"
  DIFF="$(git diff)"
  if [[ -n "$DIFF" ]]; then
    echo -e "\n=========================================================================="
    echo -e "git diff ($PWD)"
    echo -e "--------------------------------------------------------------------------"
    echo -e "$DIFF"
    echo -e "==========================================================================\n"
  fi
}

print_diff_and_env() {
  print_diff
  # Print tensorflow diff
  work_dir="$PWD"
  cd "$TF_DIR"
  print_diff
  cd "$work_dir"
  # Print env vars
  echo -e "\n=========================================================================="
  echo -e "My environment variables:"
  echo -e "--------------------------------------------------------------------------"
  printenv
  echo -e "==========================================================================\n"
}

