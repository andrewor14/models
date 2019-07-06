#!/usr/bin/env python3

from enum import Enum


# Statuses for syncing restart across replicas, state machine
# Transitions can only happen from state X to X+1, with two exceptions:
#   1. The highest state can transition back to READY_TO_SYNC, and
#   2. Any state can transition to TERMINATED
class AutoscalingStatus(Enum):
  READY_TO_SYNC = 1
  SYNCING = 2
  SYNCED = 3
  SETTING_UP = 4
  RUNNING = 5
  READY_TO_RESTART = 6
  RESTARTING = 7
  TERMINATED = -1

def get_next_status(status):
  '''
  Return the next autoscaling status assuming this process has not terminated.
  '''
  if status == AutoscalingStatus.TERMINATED:
    return status
  return AutoscalingStatus(status.value % (len(AutoscalingStatus) - 1) + 1)

# Client and server
AUTOSCALING_MASTER_HOST_PORT = "AUTOSCALING_MASTER_HOST_PORT"
AUTOSCALING_LAUNCH_WORKER_SCRIPT = "AUTOSCALING_LAUNCH_WORKER_SCRIPT"
AUTOSCALING_LAUNCH_WORKER_EVERY_N_SECONDS = "AUTOSCALING_LAUNCH_WORKER_EVERY_N_SECONDS"
AUTOSCALING_MIN_WORKERS = "AUTOSCALING_MIN_WORKERS"
AUTOSCALING_MAX_WORKERS = "AUTOSCALING_MAX_WORKERS"
AUTOSCALING_RETRY_INTERVAL_SECONDS = 1
AUTOSCALING_RPC_PORT_OFFSET = 7000

# Fake allreduce
AUTOSCALING_SAVE_LOCAL_GRADS = "AUTOSCALING_SAVE_LOCAL_GRADS"
AUTOSCALING_FAKE_ALLREDUCE_PATH = "AUTOSCALING_FAKE_ALLREDUCE_PATH"

