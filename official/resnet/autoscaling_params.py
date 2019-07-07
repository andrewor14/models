#!/usr/bin/env python3

from enum import Enum


# Statuses for syncing restart across replicas, state machine
# See `get_next_statuses` for state machine.
class AutoscalingStatus(Enum):
  READY_TO_SYNC = 1
  SYNCING = 2
  SYNCED = 3
  SETTING_UP = 4
  RUNNING = 5
  PENDING_RESTART = 6.1
  NOT_PENDING_RESTART = 6.2
  READY_TO_RESTART = 7.1
  NOT_READY_TO_RESTART = 7.2
  RESTARTING = 8
  TERMINATED = -1

def get_next_statuses(status):
  """
  Return the next autoscaling status(es) given the current status.
  """
  maybe_pending_restart = [AutoscalingStatus.PENDING_RESTART, AutoscalingStatus.NOT_PENDING_RESTART]
  maybe_ready_to_restart = [AutoscalingStatus.READY_TO_RESTART, AutoscalingStatus.NOT_READY_TO_RESTART]
  if status == AutoscalingStatus.RUNNING:
    return maybe_pending_restart
  if status in maybe_pending_restart:
    return maybe_ready_to_restart
  if status in maybe_ready_to_restart:
    return [AutoscalingStatus.RUNNING, AutoscalingStatus.RESTARTING]
  if status == AutoscalingStatus.RESTARTING:
    return [AutoscalingStatus.READY_TO_SYNC]
  if status == AutoscalingStatus.TERMINATED:
    return [status]
  return [AutoscalingStatus(status.value + 1)]

def format_statuses(statuses):
  """
  Format the given list of statuses in a nice string.
  """
  return [str(status).split(".")[-1] for status in statuses]

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

