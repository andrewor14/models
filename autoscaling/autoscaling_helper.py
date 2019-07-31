#!/usr/bin/env python3

import os
import traceback

import tensorflow as tf

from autoscaling.agent import AutoscalingAgent
from autoscaling.params import *
from autoscaling.callback import AutoscalingCallback
from autoscaling.schedule_callback import PeriodicSpawnScheduleCallback
from deploy import slurm_helper


def get_train_steps_and_epochs(num_total_samples, flags_obj, autoscaling_callback):
  """
  Return how many steps and epochs to train this round before restarting.
  """
  train_steps = num_total_samples // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs
  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, train_steps)
    train_epochs = 1
  # If we restarted in the middle of an epoch, finish the rest of the batches in the
  # epoch first, then restart again with the original number of batches in an epoch
  original_train_steps = train_steps
  original_train_epochs = train_epochs
  if autoscaling_callback.num_batches_processed_this_epoch > 0:
    train_steps -= autoscaling_callback.num_batches_processed_this_epoch
    tf.logging.info("There are %s/%s batches left in this epoch" %\
      (train_steps, original_train_steps))
    train_epochs = 1
  else:
    # Otherwise, just finish the remaining epochs
    train_epochs -= autoscaling_callback.num_epochs_processed
    tf.logging.info("There are %s/%s epochs left" %\
      (train_epochs, original_train_epochs))
  return train_steps, train_epochs

def get_schedule_callback(autoscaling_callback):
  """
  Return a `keras.callbacks.Callback` that specifies the autoscaling schedule to be used.
  """
  autoscaling_spawn_every_n_steps = int(os.getenv(AUTOSCALING_SPAWN_EVERY_N_STEPS, -1))
  autoscaling_max_workers = int(os.getenv(AUTOSCALING_MAX_WORKERS, -1))
  if autoscaling_spawn_every_n_steps > 0 and\
      autoscaling_max_workers > 0 and\
      autoscaling_callback.agent.task_index == 0:
    periodic_spawn_callback = PeriodicSpawnScheduleCallback(\
      autoscaling_callback.agent, autoscaling_spawn_every_n_steps, autoscaling_max_workers)
    periodic_spawn_callback.step_count = autoscaling_callback.num_batches_processed_this_epoch
    return periodic_spawn_callback
  return None

def run_keras(flags_obj, do_run):
  """ 
  Wrapper around main loop that handles changes in cluster membership.

  The real computation logic is specified through `do_run`, a function that takes in
  two arguments, `flags_obj` and an `AutoscalingCallback`.
  """
  # If TF_CONFIG is not provided, set it based on environment variables from slurm or MPI
  if "TF_CONFIG" not in os.environ:
    if slurm_helper.running_through_slurm():
      num_ps = int(os.getenv("NUM_PARAMETER_SERVERS", "1"))
      slurm_helper.set_tf_config(num_ps)
    elif flags_obj.use_horovod:
      from deploy import mpi_helper
      mpi_helper.set_tf_config()

  # Keep track of cluster membership changes through an autoscaling hook
  autoscaling_agent = AutoscalingAgent(flags_obj.num_gpus)
  autoscaling_callback = AutoscalingCallback(autoscaling_agent)

  while autoscaling_agent.status != AutoscalingStatus.TERMINATED:
    try:
      autoscaling_agent.initialize()
      result = do_run(flags_obj, autoscaling_callback)
      autoscaling_agent.on_restart()
      autoscaling_callback.reset()
    except Exception as e:
      tf.logging.error("Exception in resnet_main: %s (%s)" %\
        (e, e.__class__.__name__))
      traceback.print_exc()
      raise e
  return result

