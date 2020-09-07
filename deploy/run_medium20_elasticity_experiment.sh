#!/bin/bash

source common.sh

export DEBUG="true"
export FORCE_EXIT="true"
export SKIP_EVAL="true"
ORIG_LOG_DIR="$LOG_DIR"
SCHEDULER_MODE="${SCHEDULER_MODE:=WFS Priority}"
for trace_name in medium20_5jph medium20_8jph; do
  TRACE_PATH="scheduler_traces/${trace_name}.json"
  for scheduler_mode in $SCHEDULER_MODE; do
    EXPERIMENT_NAME="${trace_name}_${scheduler_mode}-scheduler"
    echo "Running trace $TRACE_PATH with $scheduler_mode scheduler"
    export LOG_DIR="${ORIG_LOG_DIR}/${EXPERIMENT_NAME}"
    mkdir -p "$LOG_DIR"
    "$PYTHON_COMMAND" -u scheduler.py "$(hostname)" 8 "$scheduler_mode"\
      "$TRACE_PATH" > "${LOG_DIR}/${EXPERIMENT_NAME}.log" 2>&1
  done
done

