#!/bin/bash

export NUM_WORKERS=4
export BATCH_SIZE=128
export NUM_EPOCHS=50
export MODE=autoscaling
export MPI_SILENCE_OUTPUT=true

# Linear function
export AUTOSCALING_UTILITY_FUNCTION_NAME=linear
#for Y_INTERCEPT in 10 100, 10000; do
for Y_INTERCEPT in 100; do
  #for PRICE in 0.01 0.1 1 10; do
  for PRICE in 0.1 1; do
    export AUTOSCALING_PRICE_PER_WORKER_PER_HOUR="$PRICE"
    export AUTOSCALING_UTILITY_FUNCTION_ARGS="$Y_INTERCEPT,10800"
    RUN_TAG="${AUTOSCALING_UTILITY_FUNCTION_NAME}"
    RUN_TAG="${RUN_TAG}-${AUTOSCALING_UTILITY_FUNCTION_ARGS}"
    RUN_TAG="${RUN_TAG}-${AUTOSCALING_PRICE_PER_WORKER_PER_HOUR}"
    export RUN_TAG
    echo "Running $RUN_TAG"
    ./run_distributed.sh
  done
done

# Step function
export AUTOSCALING_UTILITY_FUNCTION_NAME=step
for INITIAL_VALUE in 10 100 10000; do
  for PRICE in 0.01 0.1 1 10; do
    export AUTOSCALING_PRICE_PER_WORKER_PER_HOUR="$PRICE"
    export AUTOSCALING_UTILITY_FUNCTION_ARGS="$INITIAL_VALUE:3000"
    # Replace : with _, otherwise log directories will be clobbered
    FUNCTION_ARGS="${AUTOSCALING_UTILITY_FUNCTION_ARGS/:/_}"
    RUN_TAG="${AUTOSCALING_UTILITY_FUNCTION_NAME}-${FUNCTION_ARGS}"
    RUN_TAG="${RUN_TAG}-${AUTOSCALING_PRICE_PER_WORKER_PER_HOUR}"
    export RUN_TAG
    echo "Running $RUN_TAG"
    ./run_distributed.sh
  done
done

