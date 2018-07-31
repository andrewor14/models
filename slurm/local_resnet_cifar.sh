#/bin/bash

LOG_DIR="/home/andrewor/logs"
EXPR_NAME="local_resnet_cifar10"
TIMESTAMP=`date +%s`

CLUSTER="{\"ps\": [\"localhost:2222\"], \"chief\": [\"localhost:2223\"], \"worker\": [\"localhost:2224\", \"localhost:2225\"]}"

export ANDREW_RESNET_SYNC_ENABLED="true"
export ANDREW_RESNET_SYNC_AGGREGATE_REPLICAS=2
export ANDREW_RESNET_SYNC_TOTAL_REPLICAS=2

function start_it() {
  ROLE="$1"
  TASK_INDEX="$2"
  GLOBAL_INDEX="$3"
  LOG_FILE="$LOG_DIR/$EXPR_NAME-$TIMESTAMP-$GLOBAL_INDEX.out"
  export TF_CONFIG="{\"cluster\": $CLUSTER, \"task\": {\"type\": \"$ROLE\", \"index\": $TASK_INDEX}}"
  echo "Starting $ROLE-$TASK_INDEX, logging to $LOG_FILE"
  ./dist_resnet_cifar10.sh "$TIMESTAMP" > "$LOG_FILE" 2>&1 &
}

start_it "chief" 0 0
start_it "ps" 0 1
start_it "worker" 0 2
start_it "worker" 1 3

