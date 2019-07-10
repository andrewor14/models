#!/usr/bin/env python3

import os
import traceback

from autoscaling_agent import log_exceptions
from autoscaling_params import AutoscalingStatus
import tensorflow as tf


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
      run_context.session.graph._finalized = False
      self.agent.restore_variables(run_context.session)

  def do_after_run(self, run_context, run_values):
    """
    Listen for changes in cluster membership and react by restarting the server.
    """
    restarting = self.agent.step_end()
    if restarting:
      run_context.request_stop()
      if self.agent.status != AutoscalingStatus.TERMINATED:
        # If we are still training, save our variables for the next restart
        run_context.session.graph._finalized = False
        self.agent.save_variables(run_context.session)

  # ================== HELPER METHODS ==================

  def before_run(self, run_context):
    log_exceptions(lambda: self.do_before_run(run_context))

  def after_run(self, run_context, run_values):
    log_exceptions(lambda: self.do_after_run(run_context, run_values))


def log_fn(msg):
  msg = "[Autoscaling hook] %s" % msg
  tf.compat.v1.logging.info(msg)

