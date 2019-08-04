#!/usr/bin/env python3

import tensorflow as tf

from autoscaling.agent import log_exceptions
from autoscaling.params import AutoscalingStatus


class AutoscalingHook(tf.estimator.SessionRunHook):
  """
  A `SessionRunHook` that keeps track of autoscaling state for this process.
  """
  def __init__(self, agent):
    self.agent = agent

  def do_before_run(self, run_context):
    """
    Restore saved variables from memory, if any, before running the first step.
    """
    if self.agent.saved_variables is not None:
      self.agent.restore_variables(self.get_trainable_variables(), run_context.session)

  def do_after_run(self, run_context, run_values):
    """
    Listen for changes in cluster membership and react by restarting the server.
    """
    restarting = self.agent.step_end()
    if restarting:
      run_context.request_stop()

  def do_end(self, session):
    """
    Save our variables for the next restart if we are not terminating.
    """
    self.agent.train_end()
    if self.agent.status != AutoscalingStatus.TERMINATED:
      self.agent.save_variables(self.get_trainable_variables(), session)

  def get_trainable_variables(self):
    """
    Return a list of trainable variables.
    """
    return tf.global_variables()

# ================== HELPER METHODS ==================

  def before_run(self, run_context):
    log_exceptions(lambda: self.do_before_run(run_context))

  def after_run(self, run_context, run_values):
    log_exceptions(lambda: self.do_after_run(run_context, run_values))

  def end(self, session):
    log_exceptions(lambda: self.do_end(session))

def log_fn(msg):
  tf.logging.info("[Autoscaling hook] %s" % msg)

