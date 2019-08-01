#!/bin/bash

# ==================================================
#  A script for running experiments in the cluster
#
#  An experiment consists of multiple runs of a job
#  across different settings.
# ==================================================

# Set common configs
source common_configs.sh

# Configure these, see run_distributed.sh for an explanation for MODE
# These default settings will output [4, 8, 12, ... 88, 92, 96]
export MODE="${MODE:=static}"
export DATASET="${DATASET:=cifar10}"
export NUM_GPUS_INCREMENT="${NUM_GPUS_INCREMENT:=4}"
export MIN_GPUS="${MIN_GPUS:=4}"
export MAX_GPUS="${MAX_GPUS:=96}"

# Set default batch size based on dataset
if [[ "$DATASET" == "cifar10" ]]; then
  export BATCH_SIZE="${BATCH_SIZE:=128}"
elif [[ "$DATASET" == "imagenet" ]]; then
  export BATCH_SIZE="${BATCH_SIZE:=8192}"
else
  echo "ERROR: Unknown dataset '$DATASET'"
  exit 1
fi

# Don't touch these
export USE_KERAS="true"
export USE_HOROVOD="true"
export ENABLE_EAGER="true"
export NUM_PARAMETER_SERVERS="0"
export MPI_SILENCE_OUTPUT="true"

# Run the experiment
export NUM_WORKERS_LIST=`seq $MIN_GPUS $NUM_GPUS_INCREMENT $MAX_GPUS`

echo "==========================================================="
echo " Running experiment '$MODE'"
echo "   dataset = $DATASET"
echo "   batch size = $BATCH_SIZE"
echo "   num workers = "$NUM_WORKERS_LIST
echo "==========================================================="

# In static mode, we can potentially run many experiments in parallel.
# If a host file is defined, then we can use it to check which nodes are idle and
# run the remaining jobs on those nodes. Note that our approach is conservative in
# that we do not take advantage of idle *slots* (GPUs) on non-idle nodes, so as to
# avoid interfering with the network used by existing jobs.
# Note: This assumes we have SSH access to all nodes defined in the host file.
if [[ "$MODE" == "static" ]] && [[ -f "$MPI_HOST_FILE" ]]; then
  WAIT_FOR_MPI_SECONDS="5"
  RETRY_MPI_INTERVAL_SECONDS="5"

  for NUM_WORKERS in $NUM_WORKERS_LIST; do
    export NUM_WORKERS
    echo " * Searching for idle hosts to run $NUM_WORKERS workers"
    # Repeatedly try to hosts in the cluster that are not already running a job.
    # If there are such hosts, try to submit a job and observe its status.
    # If the job failed, we try again. If the job succeeded, we move on to the next job.
    while true; do
      MPI_HOSTS=""
      while read HOST; do
        # Set COLUMNS to make sure SSH output is not cut off
        RUNNING_TENSORFLOW="$(ssh -n -tt "$HOST" "COLUMNS=1000 ps aux" 2>&1 | grep run_tensorflow)"
        if [[ -z "$RUNNING_TENSORFLOW" ]]; then
          # Found an idle host, add it to MPI_HOSTS
          if [[ -z "$MPI_HOSTS" ]]; then
            MPI_HOSTS="$HOST"
          else
            MPI_HOSTS="$MPI_HOSTS,$HOST"
          fi
        fi
      done < "$MPI_HOST_FILE"

      # Try to run it, checking if the process still exists a few seconds later
      # This assumes that if there are not enough slots to run an MPI job,
      # `mpirun` will fail instantaneously.
      export MPI_HOSTS
      ./run_distributed.sh &
      RUN_PID="$!"
      sleep "$WAIT_FOR_MPI_SECONDS"
      if [[ -n "$(ps aux | grep "$RUN_PID" | grep -v 'grep')" ]]; then
        # The process is still there, so we assume it is running successfully
        echo " * Found spare capacity in the cluster, running $NUM_WORKERS workers"
        break
      fi
      # The process exited, so we assume it failed and try again later
      sleep "$RETRY_MPI_INTERVAL_SECONDS"
    done
  done

else
  # Otherwise, just launch the jobs one after another
  if [[ "$MODE" == "static" ]]; then
    echo "Warning: NOT running jobs in parallel in 'static' mode."
    echo "Alternatively, you can set MPI_HOST_FILE and this script "
    echo "will automatically find idle hosts to run the remaining "
    echo "jobs on in parallel."
  fi
  for NUM_WORKERS in $NUM_WORKERS_LIST; do
    echo " * Running with $NUM_WORKERS workers"
    export NUM_WORKERS
    ./run_distributed.sh
  done
fi

