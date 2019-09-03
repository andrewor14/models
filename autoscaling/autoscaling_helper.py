#!/usr/bin/env python3

import os
import sys
import time
import traceback

from mpi4py import MPI
import tensorflow as tf

from autoscaling.agent import AutoscalingAgent
from autoscaling.params import *
from autoscaling.callback import AutoscalingCallback
from autoscaling.schedule_callback import AutoscalingScheduleCallback
from deploy import mpi_helper
from official.utils.misc import keras_utils


# A tensorflow function that averages a list of gradients with horovod
# We refresh this function every time the cluster membership changes
# instead of rebuilding the entire graph to speed up the restart process
HOROVOD_ALLREDUCE_FUNCTION = None

# Batch sizes read by tensorflow
# Local batch size is updated on restart while global batch size is fixed
LOCAL_BATCH_SIZE = None
GLOBAL_BATCH_SIZE = None

# Step and epoch numbers read by tensorflow, updated when a worker joins
# an existing cluster and fetches the progress from the master
EPOCH_NUMBER = None
STEP_NUMBER = None

def log_fn(msg):
  tf.logging.info("[Autoscaling helper]: %s" % msg)

class BufferedIterator:
  """
  An iterator that wraps another iterator and buffers its values.

  The wrapped iterator is expected to return either a tensor or a tuple of
  tensors on each call to `next`. This is useful for changing the batch size
  for an existing iterator created from an already batched tensorflow dataset.
  """

  def __init__(self, iterator, buffer_size):
    global GLOBAL_BATCH_SIZE
    self.iterator = iterator
    self.return_buffer_size = GLOBAL_BATCH_SIZE
    self.buffer_size = buffer_size
    # A tuple of tensors, one for each item returned from `self.iterator`
    self.buf = None

  def __next__(self):
    """
    Read from the wrapped iterator, return `self.buffer_size` amount of data,
    and buffer the remaining values for the next call to `__next__`.

    The first dimension of all returned tensors is always `self.return_buffer_size`,
    padding with zeros if necessary. This allows the caller to pass the return value
    into a tensorflow function without retracing the function even when
    `self.buffer_size` changes.
    """
    while self.buf is None or self.buf[0].shape[0].value < self.buffer_size:
      next_value = next(self.iterator)
      # We assume that `self.iterator` can return a tuple of tensors.
      # If it turns out that only one tensor is returned, we force it
      # to be a tuple for simpler handling.
      if not isinstance(next_value, tuple):
        next_value = (next_value,)
      if self.buf is None:
        self.buf = next_value
      else:
        # Merge existing and new values into the respective buffer
        new_buf_values = []
        for i in range(len(self.buf)):
          new_buf_values.append(tf.concat([self.buf[i], next_value[i]], 0))
        self.buf = tuple(new_buf_values)
    # Split each buffer into data to be returned to the user this round
    # and data to be buffered for future invocations of this method
    results = []
    new_buf_values = []
    for i in range(len(self.buf)):
      results.append(self.buf[i][:self.buffer_size])
      new_buf_values.append(self.buf[i][self.buffer_size:])
    self.buf = tuple(new_buf_values)
    # Always return tensors whose first dimension is `self.return_buffer_size`,
    # filling with zeros if necessary
    for i in range(len(results)):
      num_zeros = self.return_buffer_size - results[i].shape[0].value
      zero_shape = tf.TensorShape([num_zeros] + results[i].shape.as_list()[1:])
      zeros = tf.zeros(zero_shape, dtype=results[i].dtype)
      results[i] = tf.concat([results[i], zeros], axis=0)
    return tuple(results)

  def get_next(self):
    return self.__next__()

  def set_buffer_size(self, buffer_size):
    self.buffer_size = buffer_size

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
  every_n_steps = int(os.getenv(AUTOSCALING_SPAWN_EVERY_N_STEPS, -1))
  max_workers = int(os.getenv(AUTOSCALING_MAX_WORKERS, -1))
  min_workers = int(os.getenv(AUTOSCALING_MIN_WORKERS, 1))
  spawn_size = int(os.getenv(AUTOSCALING_SPAWN_SIZE, 4))
  min_consecutive_passes_for_remove =\
    int(os.getenv(AUTOSCALING_MIN_CONSECUTIVE_PASSES_FOR_REMOVE, 3))
  min_batches_for_staying =\
    int(os.getenv(AUTOSCALING_MIN_BATCHES_FOR_STAYING, 200))
  if every_n_steps > 0 and max_workers > 0:
    return AutoscalingScheduleCallback(
      callback.agent,
      every_n_steps,
      min_workers,
      max_workers,
      spawn_size,
      min_consecutive_passes_for_remove,
      min_batches_for_staying)
  return None

def local_batch_size(global_batch_size, size, rank):
  """
  Compute this rank's local batch size.
  """
  return global_batch_size // size + int(global_batch_size % size > rank)

def initialize_horovod(comm, restarting=False):
  """
  Initialize horovod with the given communicator and set the allreduce function for
  tensorflow to call during training.
  """
  log_fn("Initializing horovod with communicator (size = %s)" % comm.size)
  import horovod.tensorflow as hvd
  if restarting:
    hvd.shutdown()
  hvd.init(comm)
  # Truncate tensor for printing
  @tf.function
  def truncate_tensor(t):
    return tf.reshape(t, [-1])[:5]
  # Allreduce function
  @tf.function
  def allreduce(grads):
    import horovod.tensorflow as hvd
    tf.logging.info("Averaging gradients with horovod (size %s)" % hvd.size())
    verbose = os.getenv("AUTOSCALING_HOROVOD_VERBOSE", "").lower() == "true"
    if verbose:
      tf.print("First gradient before horovod allreduce: ", truncate_tensor(grads[0]))
    grads = [hvd.allreduce(grad) for grad in grads]
    if verbose:
      tf.print("First gradient after horovod allreduce: ", truncate_tensor(grads[0]))
    return grads
  global HOROVOD_ALLREDUCE_FUNCTION
  HOROVOD_ALLREDUCE_FUNCTION = allreduce

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
  agent = AutoscalingAgent(flags_obj.num_gpus, flags_obj.batch_size, flags_obj.use_horovod)
  callback = AutoscalingCallback(agent)

  # TODO: no need for this loop anymore?
  while agent.status != AutoscalingStatus.TERMINATED:
    agent.initialize()
    if flags_obj.use_horovod:
      initialize_horovod(agent.mpi_communicator)
    # Actually run the training
    # We expect this function to call model.fit
    result = do_run(flags_obj, callback)
    agent.on_restart()
    callback.reset()
    if flags_obj.use_horovod:
      import horovod.tensorflow as hvd
      hvd.shutdown()

  # Make sure everyone exits together to avoid connection refused errors
  if is_checkpoint_restart_mode():
    agent.mpi_communicator.barrier()

  log_fn("Training complete")

  # Hack: exit only if master is dead
  # If we exit early then we may trigger MPI errors that will fail the whole job
  if agent.mpi_communicator.rank != 0:
    try:
      while True:
        agent.client.master_server_rpc(lambda s: s.get_status())
        time.sleep(AUTOSCALING_EXIT_RETRY_INTERVAL_SECONDS)
    except (ConnectionRefusedError, ConnectionResetError) as e:
      pass
  else:
    MPI.COMM_WORLD.Abort()
  log_fn("Exiting process")

