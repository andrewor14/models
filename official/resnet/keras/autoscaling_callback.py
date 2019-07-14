#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.python import keras

from official.resnet.autoscaling_agent import log_exceptions
from official.resnet.autoscaling_params import AutoscalingStatus


class AutoscalingCallback(keras.callbacks.Callback):
  """
  A `keras.callbacks.Callback` that keeps track of autoscaling state for this process.
  """
  def __init__(self, agent):
    self.agent = agent
    self.model = None
    self.num_batches_processed_this_epoch = 0
    self.num_epochs_processed = 0
    # Run this callback on all the workers
    self._chief_worker_only = False

  def set_model(self, model):
    self.model = model

  def reset(self):
    self.model = None

  def do_on_batch_begin(self, batch, logs):
    """
    Restore saved variables from memory, if any, before running the first step.
    """
    if self.agent.saved_variables is not None:
      self.agent.restore_variables(self.get_trainable_variables())

  def do_on_batch_end(self, batch, logs):
    """
    Listen for changes in cluster membership and react by restarting the server.
    """
    self.num_batches_processed_this_epoch += 1
    restarting = self.agent.step_end()
    if restarting:
      self.model.stop_training = True
      if self.agent.status != AutoscalingStatus.TERMINATED:
        # If we are still training, save our variables for the next restart
        self.agent.save_variables(self.get_trainable_variables())

  def do_on_epoch_end(self, epoch, logs):
    """
    Record the number of epochs processed so far.
    """
    # This is still called when we're restarting, but that shouldn't count as an epoch end
    if self.agent.status == AutoscalingStatus.RUNNING:
      self.num_epochs_processed += 1
      self.num_batches_processed_this_epoch = 0

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

  def on_epoch_end(self, epoch, logs=None):
    log_exceptions(lambda: self.do_on_epoch_end(epoch, logs))

def log_fn(msg):
  msg = "[Autoscaling callback] %s" % msg
  tf.compat.v1.logging.info(msg)

