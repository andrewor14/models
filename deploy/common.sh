#!/bin/bash

export ENVIRONMENT="$(hostname | awk -F '[.-]' '{print $1}' | sed 's/[0-9]//g')"

if [[ "$ENVIRONMENT" == "snsgpu" ]]; then
  export BASE_DIR="/home/andrew/Documents/dev"
  export PYTHON_COMMAND="/usr/bin/python3"
  export DEFAULT_NUM_GPUS_PER_NODE="2"
elif [[ "$ENVIRONMENT" == "ns" ]]; then
  export BASE_DIR="/home/andrewor"
  export PYTHON_COMMAND="/usr/licensed/anaconda3/5.2.0/bin/python3.6"
  export NUM_GPUS="0"
  export DEFAULT_NUM_GPUS_PER_NODE="1"
elif [[ -n "$IN_DOCKER_CONTAINER" ]]; then
  export ENVIRONMENT="docker"
  export BASE_DIR="/root/dev"
  export PYTHON_COMMAND="/usr/bin/python3"
  export DEFAULT_NUM_GPUS_PER_NODE="8"
else
  echo "ERROR: Unknown environment '$ENVIRONMENT'"
  exit 1
fi
export TIMESTAMP=`date +%m_%d_%y_%s%3N`
export TF_DIR="$BASE_DIR/tensorflow"
export LOG_DIR="${LOG_DIR:=$BASE_DIR/logs}"
export MODELS_DIR="$BASE_DIR/models"
export PYTHONPATH="$PYTHONPATH:$MODELS_DIR"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:=$DEFAULT_NUM_GPUS_PER_NODE}"

# Tensorflow related flags
export NUM_GPUS="${NUM_GPUS:=1}"
export NUM_NODES="${NUM_NODES:=1}"
export DTYPE="${DTYPE:=fp16}"
export LOG_STEPS="${LOG_STEPS:=1}"
export SKIP_EVAL="${SKIP_EVAL:=false}"
export NUM_VIRTUAL_NODES_PER_DEVICE="${NUM_VIRTUAL_NODES_PER_DEVICE:=1}"
export SAVED_CHECKPOINT_PATH="${SAVED_CHECKPOINT_PATH:=}"
export ENABLE_CHECKPOINTS="${ENABLE_CHECKPOINTS:=false}"
export NUM_CHECKPOINTS_TO_KEEP="${NUM_CHECKPOINTS_TO_KEEP:=5}"
export ENABLE_ELASTICITY="${ENABLE_ELASTICITY:=false}"

# Monitor memory
export ENABLE_MONITOR_MEMORY="${ENABLE_MONITOR_MEMORY:=false}"
if [[ "$ENABLE_MONITOR_MEMORY" == "true" ]]; then
  # Force GPU memory to grow if we're monitoring it
  export TF_FORCE_GPU_ALLOW_GROWTH="true"
fi

# Log memory
export LOG_MEMORY_ENABLED="${LOG_MEMORY_ENABLED:=false}"
if [[ "$LOG_MEMORY_ENABLED" == "true" ]]; then
  export TF_CPP_MIN_VLOG_LEVEL="1"
fi

# XLA
DEFAULT_ENABLE_XLA="true"
if [[ "$ENABLE_ELASTICITY" == "true" ]]; then
  DEFAULT_ENABLE_XLA="false"
fi
export ENABLE_XLA="${ENABLE_XLA:=$DEFAULT_ENABLE_XLA}"

# Set distribution strategy
if [[ "$ENABLE_ELASTICITY" == "true" ]] || [[ "$NUM_NODES" == "1" ]]; then
  export DEFAULT_DISTRIBUTION_STRATEGY="mirrored"
else
  export DEFAULT_DISTRIBUTION_STRATEGY="multi_worker_mirrored"
fi
export DISTRIBUTION_STRATEGY="${DISTRIBUTION_STRATEGY:=$DEFAULT_DISTRIBUTION_STRATEGY}"

# Optionally use a process' rank as CUDA_VISIBLE_DEVICES
if [[ "$USE_RANK_FOR_CVD" == "true" ]]; then
  export CUDA_VISIBLE_DEVICES="${SPAWN_RANK:-$OMPI_COMM_WORLD_RANK}"
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
  # Optionally print current python processes
  if [[ "$PRINT_PROCESSES" == "true" ]]; then
    echo -e "\n=========================================================================="
    echo -e "Current running processes:"
    echo -e "--------------------------------------------------------------------------"
    ps aux | grep 'python\|run_distributed'
    echo -e "--------------------------------------------------------------------------"
    nvidia-smi
    echo -e "==========================================================================\n"
  fi
}

# If this is a spawned process, set the log file accordingly
maybe_set_spawn_log_file() {
  if [[ -n "$SPAWN_RANK" ]]; then
    LOG_DIR="${LOG_DIR}/${JOB_NAME}/1/rank.${SPAWN_RANK}"
    mkdir -p "$LOG_DIR"
    export LOG_FILE="${LOG_DIR}/stderr"
    # If the log file already exists, append a .x to the file name
    i=1
    while [[ -f "$LOG_FILE" ]]; do
      export LOG_FILE="${LOG_DIR}/stderr.${i}"
      i="$((i+1))"
    done
  fi
}

