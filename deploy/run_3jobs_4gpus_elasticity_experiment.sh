#!/bin/bash

source common.sh

export DEBUG="true"
export FORCE_EXIT="true"
ORIG_LOG_DIR="$LOG_DIR"
for version in 1 2 3; do
  TRACE_NAME="3jobs_4gpus_v${version}"
  TRACE_PATH="scheduler_traces/${TRACE_NAME}.json"
  for scheduler_mode in "WFS" "Priority"; do
    EXPERIMENT_NAME="${TRACE_NAME}_${scheduler_mode}-scheduler"
    echo "Running trace $TRACE_PATH with $scheduler_mode scheduler"
    export LOG_DIR="${ORIG_LOG_DIR}/${EXPERIMENT_NAME}"
    mkdir -p "$LOG_DIR"
    "$PYTHON_COMMAND" -u scheduler.py "$(hostname)" 4 "$scheduler_mode"\
      "$TRACE_PATH" > "${LOG_DIR}/${EXPERIMENT_NAME}.log" 2>&1
  done
done

