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
    # Run this callback on all the workers
    self._chief_worker_only = False

  def do_on_batch_begin(self, batch, logs):
    """
    Restore saved variables from memory, if any, before running the first step.
    """
    log_fn("on batch begin")
    if self.agent.saved_variables is not None:
      # TODO: restore variables
      pass

  def do_on_batch_end(self, batch, logs):
    """
    Listen for changes in cluster membership and react by restarting the server.
    """
    restarting = self.agent.step_end()
    if restarting:
      self.model.stop_training = True
      if self.agent.status != AutoscalingStatus.TERMINATED:
        # If we are still training, save our variables for the next restart
        # TODO: save variables
        pass

  # ================== HELPER METHODS ==================

  def on_batch_begin(self, batch, logs=None):
    log_exceptions(lambda: self.do_on_batch_begin(batch, logs))

  def on_batch_end(self, batch, logs=None):
    log_exceptions(lambda: self.do_on_batch_end(batch, logs))


def log_fn(msg):
  msg = "[Autoscaling callback] %s" % msg
  tf.compat.v1.logging.info(msg)

