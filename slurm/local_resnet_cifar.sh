#/bin/bash

LOG_DIR="/home/andrewor/logs"
EXPR_NAME="local_resnet_cifar10"
TIMESTAMP=`date +%s`

# Number of processes: 1 parameter server, 1 chief, (n-2) workers
NUM_PROCESSES="${1:-4}"
if [[ "$NUM_PROCESSES" < 3 ]]; then
  echo "Number of processes must be >= 3 (was $NUM_PROCESSES)"
  exit 1
fi
NUM_WORKERS=$(($NUM_PROCESSES - 2))

# Enable k-sync mode
export RESNET_K_SYNC_ENABLED="true"
export RESNET_K_SYNC_STARTING_AGGREGATE_REPLICAS=1
export RESNET_K_SYNC_TOTAL_REPLICAS="$NUM_WORKERS"

# Build the json string for ClusterSpec, to be used in TF_CONFIG
function build_tf_config() {
  CLUSTER="{\"ps\": [\"localhost:2222\"], \"chief\": [\"localhost:2223\"], \"worker\": ["
  STARTING_PORT=2223
  for i in `seq 1 $NUM_WORKERS`; do
    PORT=$(($STARTING_PORT + $i))
    CLUSTER="$CLUSTER\"localhost:$PORT\""
    if [[ "$i" < "$NUM_WORKERS" ]]; then
      CLUSTER="$CLUSTER, "
    fi
  done
  CLUSTER="$CLUSTER]}"
}

# Start a process in the background and redirect everything to the appropriate log file
function start_it() {
  ROLE="$1"
  TASK_INDEX="$2"
  GLOBAL_INDEX="$3"
  LOG_FILE="$LOG_DIR/$EXPR_NAME-$TIMESTAMP-$GLOBAL_INDEX.out"
  export TF_CONFIG="{\"cluster\": $CLUSTER, \"task\": {\"type\": \"$ROLE\", \"index\": $TASK_INDEX}}"
  echo "Starting $ROLE-$TASK_INDEX, logging to $LOG_FILE"
  ./dist_resnet_cifar10.sh "$TIMESTAMP" > "$LOG_FILE" 2>&1 &
}

# Actually start everything
build_tf_config
echo "ClusterSpec: $CLUSTER"
start_it "chief" 0 0
start_it "ps" 0 1
for i in `seq 0 $(($NUM_WORKERS - 1))`; do
  start_it "worker" $i $(($i + 2))
done

