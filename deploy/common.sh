#!/bin/bash

export ENVIRONMENT="$(hostname | awk -F '[.-]' '{print $1}' | sed 's/[0-9]//g')"

if [[ "$ENVIRONMENT" == "snsgpu" ]]; then
  export BASE_DIR="/home/andrew/Documents/dev"
  export PYTHON_COMMAND="/usr/bin/python3"
elif [[ "$ENVIRONMENT" == "ns" ]]; then
  export BASE_DIR="/home/andrewor"
  export PYTHON_COMMAND="/usr/licensed/anaconda3/5.2.0/bin/python3.6"
  export NUM_GPUS="0"
elif [[ -n "$IN_DOCKER_CONTAINER" ]]; then
  export ENVIRONMENT="docker"
  export BASE_DIR="/root/dev"
  export PYTHON_COMMAND="/usr/bin/python3"
else
  echo "ERROR: Unknown environment '$ENVIRONMENT'"
  exit 1
fi
export TIMESTAMP=`date +%m_%d_%y_%s%3N`
export TF_DIR="$BASE_DIR/tensorflow"
export LOG_DIR="${LOG_DIR:=$BASE_DIR/logs}"
export PYTHONPATH="$PYTHONPATH:$BASE_DIR/models"

# Tensorflow related flags
export DTYPE="${DTYPE:=fp16}"
export LOG_STEPS="${LOG_STEPS:=1}"
export SKIP_EVAL="${SKIP_EVAL:=false}"
export NUM_VIRTUAL_NODES_PER_DEVICE="${NUM_VIRTUAL_NODES_PER_DEVICE:=1}"

export ENABLE_XLA="${ENABLE_XLA:=false}"
if [[ "$ENABLE_XLA" == "true" ]]; then
  export TF_XLA_FLAGS="${TF_XLA_FLAGS:=--tf_xla_cpu_global_jit}"
fi

export LOG_MEMORY_ENABLED="${LOG_MEMORY_ENABLED:=false}"
if [[ "$LOG_MEMORY_ENABLED" == "true" ]]; then
  export TF_CPP_MIN_VLOG_LEVEL="1"
fi

# Set `JOB_NAME` to a unique, identifiable value
set_job_name() {
  JOB_NAME="$1"
  if [[ -n "$RUN_TAG" ]]; then
    JOB_NAME="${JOB_NAME}-${RUN_TAG}"
  fi
  export JOB_NAME="${JOB_NAME}-${TIMESTAMP}"
}

# Print the output of `git diff` in the current working directory
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

# Print the values of existing environment variables and the output of `git diff`
# in both the `models` and the `tensorflow` repositories
print_diff_and_env() {
  print_diff
  # Print tensorflow diff
  if [[ -d "$TF_DIR" ]]; then
    work_dir="$PWD"
    cd "$TF_DIR"
    print_diff
    cd "$work_dir"
  fi
  # Print env vars
  echo -e "\n=========================================================================="
  echo -e "My environment variables:"
  echo -e "--------------------------------------------------------------------------"
  printenv
  echo -e "==========================================================================\n"
}

