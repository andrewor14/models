#!/bin/bash

export NUM_WORKERS=1
export BATCH_SIZE=1024
export NUM_EPOCHS=100
export AUTOSCALING_HOROVOD_VERBOSE=true
export AUTOSCALING_STRAGGLER_RANKS=3,7,11,15,19,23,27,31,35,39
export AUTOSCALING_DETACH_WHEN_REMOVED=false
export MPI_SILENCE_OUTPUT=true

# Always increase
export AUTOSCALING_THROUGHPUT_SCALING_THRESHOLD=-100000
export AUTOSCALING_SPAWN_SIZE=1
export AUTOSCALING_MIN_WORKERS=1
export AUTOSCALING_MAX_WORKERS=50

for multiplier in 1 1.333333 2 4; do
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

