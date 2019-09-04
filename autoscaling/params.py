#!/usr/bin/env python3

from enum import Enum
import os


# Client and server
AUTOSCALING_MASTER_HOST_PORT = "AUTOSCALING_MASTER_HOST_PORT"
AUTOSCALING_RETRY_INTERVAL_SECONDS = 0.1
AUTOSCALING_JOIN_RETRY_INTERVAL_SECONDS = 1
AUTOSCALING_RPC_PORT_OFFSET = 7000

# Scaling
AUTOSCALING_MIN_WORKERS = "AUTOSCALING_MIN_WORKERS"
AUTOSCALING_MAX_WORKERS = "AUTOSCALING_MAX_WORKERS"
AUTOSCALING_INITIAL_WORKERS = "AUTOSCALING_INITIAL_WORKERS"
AUTOSCALING_SPAWN_EVERY_N_STEPS = "AUTOSCALING_SPAWN_EVERY_N_STEPS"
AUTOSCALING_SPAWN_SIZE = "AUTOSCALING_SPAWN_SIZE"
AUTOSCALING_MIN_STEPS_BETWEEN_RESTART = "AUTOSCALING_MIN_STEPS_BETWEEN_RESTART"
AUTOSCALING_THROUGHPUT_SCALING_THRESHOLD = "AUTOSCALING_THROUGHPUT_SCALING_THRESHOLD"
AUTOSCALING_MIN_CONSECUTIVE_PASSES_FOR_REMOVE = "AUTOSCALING_MIN_CONSECUTIVE_PASSES_FOR_REMOVE"
AUTOSCALING_MIN_BATCHES_FOR_STAYING = "AUTOSCALING_MIN_BATCHES_FOR_STAYING"

# Utility functions
AUTOSCALING_UTILITY_FUNCTION_NAME = "AUTOSCALING_UTILITY_FUNCTION_NAME"
AUTOSCALING_UTILITY_FUNCTION_ARGS = "AUTOSCALING_UTILITY_FUNCTION_ARGS"
AUTOSCALING_UTILITY_FUNCTION_LINEAR = "linear"
AUTOSCALING_UTILITY_FUNCTION_STEP = "step"
AUTOSCALING_PRICE_PER_WORKER_PER_SECOND = "AUTOSCALING_PRICE_PER_WORKER_PER_SECOND"

# Checkpoint restart
AUTOSCALING_CHECKPOINT_DIR = "AUTOSCALING_CHECKPOINT_DIR"
AUTOSCALING_CHECKPOINT_FILE_NAME = "checkpoint.h5"
AUTOSCALING_CHECKPOINT_METADATA_FILE_NAME = "checkpoint.metadata"
AUTOSCALING_SCHEDULE_THROUGHPUT_PREFIX = "AUTOSCALING_SCHEDULE_THROUGHPUT_"
AUTOSCALING_SCHEDULE_TOTAL_TIME = "AUTOSCALING_SCHEDULE_TOTAL_TIME"

# Other
AUTOSCALING_LOCAL_BATCH_SIZE = "AUTOSCALING_LOCAL_BATCH_SIZE"
AUTOSCALING_NUM_BATCHES_PROCESSED_THIS_EPOCH = "AUTOSCALING_NUM_BATCHES_PROCESSED_THIS_EPOCH"
AUTOSCALING_NUM_EPOCHS_PROCESSED = "AUTOSCALING_NUM_EPOCHS_PROCESSED"
AUTOSCALING_REPLACE_STRAGGLERS = "AUTOSCALING_REPLACE_STRAGGLERS"
AUTOSCALING_STRAGGLER_RANKS = "AUTOSCALING_STRAGGLER_RANKS"
AUTOSCALING_STRAGGLER_MULTIPLIER = "AUTOSCALING_STRAGGLER_MULTIPLIER"
AUTOSCALING_STRAGGLER_THRESHOLD = "AUTOSCALING_STRAGGLER_THRESHOLD"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
ORIGINAL_CUDA_VISIBLE_DEVICES = "ORIGINAL_CUDA_VISIBLE_DEVICES"
TENSORFLOW_COMPUTE_ELAPSED = "compute_elapsed"

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

def is_autoscaling_mode():
  return get_autoscaling_mode() == AutoscalingMode.AUTOSCALING

# Statuses for syncing restart across replicas, state machine
# See `get_next_statuses` for state machine.
class AutoscalingStatus(Enum):
  READY_TO_SYNC = 1
  SYNCING = 2
  SYNCED = 3
  RUNNING = 4
  RESTARTING = 5
  TERMINATED = -1

def is_syncing(status):
  """
  Return whether the given status represents one that is syncing cluster specs.
  """
  return status in [
    AutoscalingStatus.READY_TO_SYNC,
    AutoscalingStatus.SYNCING,
    AutoscalingStatus.SYNCED]

