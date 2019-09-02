#!/bin/bash

export NUM_WORKERS=1
export BATCH_SIZE=1024
export AUTOSCALING_HOROVOD_VERBOSE=true
export AUTOSCALING_SCHEDULE=linear_increase
export AUTOSCALING_STRAGGLER_RANKS=2,5,8,11,14,17,20,23,26,29

for multiplier in 1.333333 2 4; do
  export AUTOSCALING_STRAGGLER_MULTIPLIER="$multiplier"
  for replace in "true" "false"; do
    export AUTOSCALING_REPLACE_STRAGGLERS="$replace"
    export RUN_TAG="straggler-$multiplier"
    if [[ "$AUTOSCALING_REPLACE_STRAGGLERS" == "true" ]]; then
      export RUN_TAG="$RUN_TAG-replace"
    fi
    echo "$RUN_TAG"
    ./run_distributed.sh
  done
done

