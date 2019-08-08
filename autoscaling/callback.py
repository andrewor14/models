#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.python import keras

from autoscaling.agent import log_exceptions
from autoscaling.params import AutoscalingStatus


class AutoscalingCallback(keras.callbacks.Callback):
  """
  A `keras.callbacks.Callback` that keeps track of autoscaling state for this process.
  """
  def __init__(self, agent):
    self.agent = agent
    self.model = None
    self.num_batches_per_epoch = None
    self.num_batches_processed_this_epoch = 0
    self.num_epochs_processed = 0
    # Run this callback on all the workers
    self._chief_worker_only = False
    # Expose our progress method to the autoscaling service through our agent
    self.agent.get_progress_method = self.get_progress
    self.bootstrap_progress()

  def set_model(self, model):
    self.model = model

  def reset(self):
    self.model = None

  def get_progress(self):
    """
    Return a 2-tuple of
      (1) Number of batches processed in this epoch so far, and
      (2) Number of epochs processed so far.
    """
    return (self.num_batches_processed_this_epoch, self.num_epochs_processed)

  def bootstrap_progress(self):
    """
    Bootstrap this worker so it can start on the same step as everyone else.

    We do this by fetching the progress from the master autoscaling server.
    Note: we must do this after the master is READY_TO_SYNC, otherwise the
    progress may be wrong if the master is still running.
    """
    self.agent.status_barrier(AutoscalingStatus.READY_TO_SYNC)
    num_batches_processed_this_epoch, num_epochs_processed =\
      self.agent.client.master_server.get_progress()
    if num_batches_processed_this_epoch is not None and num_epochs_processed is not None:
      log_fn("Fetched progress from master server = (%s steps, %s epochs)" %\
        (num_batches_processed_this_epoch, num_epochs_processed))
      self.num_batches_processed_this_epoch = num_batches_processed_this_epoch
      self.num_epochs_processed = num_epochs_processed
    else:
      log_fn("Warning: unable to fetch progress from master server")

  def do_on_batch_begin(self, batch, logs):
    """
    Restore saved variables from memory, if any, before running the first step.
    """
    if self.agent.saved_variables is not None:
      #self.agent.restore_variables(self.get_trainable_variables())
      pass

  def do_on_batch_end(self, batch, logs):
    """
    Listen for changes in cluster membership and react by restarting the server.
    """
    # Update counters
    self.num_batches_processed_this_epoch += 1
    if self.num_batches_processed_this_epoch == self.num_batches_per_epoch:
      self.num_epochs_processed += 1
      self.num_batches_processed_this_epoch = 0
    # Check if we need to restart
    restarting = self.agent.step_end()
    if restarting:
      self.model.stop_training = True

  def do_on_train_end(self, logs):
    """
    Save our variables for the next restart if we are not terminating.
    """
    if self.agent.status != AutoscalingStatus.TERMINATED:
      self.agent.save_variables(self.get_trainable_variables())

  def get_trainable_variables(self):
    """
    Return a list of trainable variables.
    """
    if self.model is None:
      raise ValueError("Model must be set before the first step.")
    return self.model.trainable_variables

  # ================== HELPER METHODS ==================

  def on_batch_begin(self, batch, logs=None):
    log_exceptions(lambda: self.do_on_batch_begin(batch, logs))

  def on_batch_end(self, batch, logs=None):
    log_exceptions(lambda: self.do_on_batch_end(batch, logs))

  def on_train_end(self, logs=None):
    log_exceptions(lambda: self.do_on_train_end(logs))

def log_fn(msg):
  tf.logging.info("[Autoscaling callback] %s" % msg)

