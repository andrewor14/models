#!/usr/bin/env python3

import os
import sys
import traceback

from mpi4py import MPI
import tensorflow as tf

from autoscaling.agent import AutoscalingAgent
from autoscaling.params import *
from autoscaling.callback import AutoscalingCallback
from autoscaling.schedule_callback import PeriodicSpawnScheduleCallback
from deploy import mpi_helper
from official.utils.misc import keras_utils


def log_fn(msg):
  tf.logging.info("[Autoscaling helper]: %s" % msg)

def get_train_steps_and_epochs(num_total_samples, flags_obj, callback):
  """
  Return how many steps and epochs to train this round before restarting.
  """
  train_steps = num_total_samples // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs
  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, train_steps)
    train_epochs = 1
  callback.num_batches_per_epoch = train_steps
  callback.num_epochs_total = train_epochs
  # If we restarted in the middle of an epoch, finish the rest of the batches in the
  # epoch first, then restart again with the original number of batches in an epoch
  original_train_steps = train_steps
  original_train_epochs = train_epochs
  if callback.num_batches_processed_this_epoch > 0:
    train_steps -= callback.num_batches_processed_this_epoch
    tf.logging.info("There are %s/%s batches left in this epoch" %\
      (train_steps, original_train_steps))
    train_epochs = 1
  else:
    # Otherwise, just finish the remaining epochs
    train_epochs -= callback.num_epochs_processed
    tf.logging.info("There are %s/%s epochs left" %\
      (train_epochs, original_train_epochs))
  return train_steps, train_epochs

def get_schedule_callback(callback):
  """
  Return a `keras.callbacks.Callback` that specifies the autoscaling schedule to be used.
  """
  autoscaling_spawn_every_n_steps = int(os.getenv(AUTOSCALING_SPAWN_EVERY_N_STEPS, -1))
  autoscaling_max_workers = int(os.getenv(AUTOSCALING_MAX_WORKERS, -1))
  if autoscaling_spawn_every_n_steps > 0 and\
      autoscaling_max_workers > 0 and\
      callback.agent.task_index == 0:
    periodic_spawn_callback = PeriodicSpawnScheduleCallback(\
      callback.agent, autoscaling_spawn_every_n_steps, autoscaling_max_workers)
    periodic_spawn_callback.step_count = callback.num_batches_processed_this_epoch
    return periodic_spawn_callback
  return None

def initialize_horovod(flags_obj, comm):
  import horovod.tensorflow as hvd
  # HACK: Horovod freezes when restarting with a larger communicator.
  # However, this issue goes away if we first restart with MPI.COMM_WORLD
  # and also clear any lingering state in tensorflow's keras backend.
  # Admittedly, it is unclear why this works, but it does!
  tf.keras.backend.clear_session()
  hvd.init(MPI.COMM_WORLD.Dup())
  hvd.allreduce(tf.constant(hvd.rank()))
  hvd.shutdown()
  tf.keras.backend.clear_session()
  hvd.init(comm)

def run_keras(flags_obj, do_run):
  """ 
  Wrapper around main loop that handles changes in cluster membership.

  The real computation logic is specified through `do_run`, a function that takes in
  two arguments, `flags_obj` and an `AutoscalingCallback`.

  WARNING: this function currently terminates the python process on finish or error.
  """
  # Set TF_CONFIG using MPI
  if "TF_CONFIG" not in os.environ:
    mpi_helper.set_tf_config()

  # Always enable eager execution in the beginning
  keras_utils.set_session_config(
    enable_eager=flags_obj.enable_eager,
    enable_xla=flags_obj.enable_xla)

  # Keep track of cluster membership changes through an autoscaling hook
  agent = AutoscalingAgent(flags_obj.num_gpus, flags_obj.use_horovod)
  callback = AutoscalingCallback(agent)

  while agent.status != AutoscalingStatus.TERMINATED:
    try:
      agent.initialize()
      if flags_obj.use_horovod:
        initialize_horovod(flags_obj, agent.mpi_communicator)
      # Actually run the training
      # We expect this function to call model.fit
      result = do_run(flags_obj, callback)
      agent.on_restart()
      callback.reset()
      if flags_obj.use_horovod:
        import horovod.tensorflow as hvd
        hvd.shutdown()
    except Exception as e:
      tf.logging.error("Exception in resnet_main: %s (%s)" %\
        (e, e.__class__.__name__))
      traceback.print_exc()
      # Hack: the tensorflow process does not terminate properly unless we do this
      os._exit(1)

  log_fn("Training complete")
  # Hack: the tensorflow process does not terminate properly unless we do this
  os._exit(0)

