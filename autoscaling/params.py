#!/usr/bin/env python3

from enum import Enum
import os


# Client and server
AUTOSCALING_MASTER_HOST_PORT = "AUTOSCALING_MASTER_HOST_PORT"
AUTOSCALING_RETRY_INTERVAL_SECONDS = 1
AUTOSCALING_RPC_PORT_OFFSET = 7000

# Scaling
AUTOSCALING_MIN_WORKERS = "AUTOSCALING_MIN_WORKERS"
AUTOSCALING_MAX_WORKERS = "AUTOSCALING_MAX_WORKERS"
AUTOSCALING_SPAWN_EVERY_N_STEPS = "AUTOSCALING_SPAWN_EVERY_N_STEPS"

# Checkpoint restart
AUTOSCALING_CHECKPOINT_DIR = "AUTOSCALING_CHECKPOINT_DIR"
AUTOSCALING_CHECKPOINT_FILE_NAME = "checkpoint.h5"
AUTOSCALING_CHECKPOINT_METADATA_FILE_NAME = "checkpoint.metadata"

# Other
AUTOSCALING_NUM_BATCHES_PROCESSED_THIS_EPOCH = "AUTOSCALING_NUM_BATCHES_PROCESSED_THIS_EPOCH"
AUTOSCALING_NUM_EPOCHS_PROCESSED = "AUTOSCALING_NUM_EPOCHS_PROCESSED"
AUTOSCALING_SYNC_INTERVAL_STEPS = "AUTOSCALING_SYNC_INTERVAL_STEPS"

# Mode to identify whether and how scaling is done
class AutoscalingMode(Enum):
  STATIC = "static"
  CHECKPOINT_RESTART = "checkpoint-restart"
  AUTOSCALING = "autoscaling"

def get_autoscaling_mode():
  mode = os.environ["MODE"]
  if mode not in AutoscalingMode._value2member_map_:
    raise ValueError("Unknown autoscaling mode '%s'" % mode)
  return AutoscalingMode._value2member_map_[mode]

def is_checkpoint_restart_mode():
  return get_autoscaling_mode() == AutoscalingMode.CHECKPOINT_RESTART

# Statuses for syncing restart across replicas, state machine
# See `get_next_statuses` for state machine.
class AutoscalingStatus(Enum):
  READY_TO_SYNC = 1
  SYNCING = 2
  SYNCED = 3
  RUNNING = 4
  PENDING_RESTART = 5.1
  NOT_PENDING_RESTART = 5.2
  READY_TO_RESTART = 6.1
  NOT_READY_TO_RESTART = 6.2
  RESTARTING = 7
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
    return [AutoscalingStatus.RUNNING, AutoscalingStatus.RESTARTING, AutoscalingStatus.TERMINATED]
  if status == AutoscalingStatus.RESTARTING:
    return [AutoscalingStatus.READY_TO_SYNC]
  if status == AutoscalingStatus.TERMINATED:
    return [status]
  return [AutoscalingStatus(status.value + 1)]

def is_running(status):
  """
  Return whether the given status represents one that is running and not pending to restart.
  """
  return status in [
    AutoscalingStatus.RUNNING,
    AutoscalingStatus.NOT_PENDING_RESTART,
    AutoscalingStatus.NOT_READY_TO_RESTART]

def is_syncing(status):
  """
  Return whether the given status represents one that is syncing cluster specs.
  """
  return status in [
    AutoscalingStatus.READY_TO_SYNC,
    AutoscalingStatus.SYNCING,
    AutoscalingStatus.SYNCED]

def format_statuses(statuses):
  """
  Format the given list of statuses in a nice string.
  """
  return [str(status).split(".")[-1] for status in statuses]

