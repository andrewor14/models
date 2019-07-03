#!/usr/bin/env python3

import os
import json
import threading

import tensorflow as tf

from autoscaling_client import convert_port, AutoscalingClient
from autoscaling_service import listen_for_autoscaling_requests
from autoscaling_params import *


def log_fn(msg):
  msg = "[Autoscaling hook] %s" % msg
  tf.compat.v1.logging.info(msg)

def get_tf_config():
  """
  Return TF_CONFIG as a dictionary, assuming it was set.
  """
  tf_config = os.getenv("TF_CONFIG")
  if tf_config is None:
    raise ValueError("'TF_CONFIG' must be set.")
  return json.loads(tf_config.replace("'", "\""))


class AutoscalingHook(tf.estimator.SessionRunHook):
  """
  A `SessionRunHook` that keeps track of autoscaling state for this process.
  """

  def __init__(self, estimator):
    self.estimator = estimator
    self.server = None
    self.global_batch_size = None

    # Status to synchronize cluster membership changes
    # Accesses must be guarded by `self._autoscaling_status_lock`
    self._autoscaling_status = AutoscalingStatus.READY_TO_SYNC
    self._autoscaling_status_lock = threading.Lock()

    # The cluster spec to apply next time we reinitialize
    # Note: We can't apply the new cluster spec right away because doing so will
    # signal to other workers that they can go ahead and build their graphs, which
    # might be inconsistent with ours if we haven't finished reinitializing yet.
    # Accesses must be guarded by `self.pending_cluster_spec_lock`.
    self.pending_cluster_spec = None
    self.pending_cluster_spec_lock = threading.Lock()    

    # Parse this process' host port from TF_CONFIG
    tf_config = get_tf_config()
    task_type = tf_config["task"]["type"]
    task_index = tf_config["task"]["index"]
    self.cluster_spec = tf_config["cluster"]
    self.host_port = self.cluster_spec[task_type][task_index]

    # Start autoscaling server
    listen_for_autoscaling_requests(self, convert_port(self.host_port))

    # Start autoscaling client, connected to the autoscaling server on the first worker
    first_worker = os.getenv(AUTOSCALING_MASTER_HOST_PORT)
    if first_worker is None:
      first_worker = tf_config["cluster"]["worker"][0]
      first_worker = convert_port(first_worker)
    self.autoscaling_client = AutoscalingClient(first_worker)

    # Request to join the cluster
    self.autoscaling_client.master_server.join_cluster(self.host_port)

  @property
  def autoscaling_status(self):
    """
    Atomic getter for `self._autoscaling_status`, guarded by `self._autoscaling_status_lock`.
    """
    try:
      self._autoscaling_status_lock.acquire()
      return self._autoscaling_status
    finally:
      self._autoscaling_status_lock.release()

  @autoscaling_status.setter
  def autoscaling_status(self, status):
    """
    Atomic setter for `self._autoscaling_status`, guarded by `self._autoscaling_status_lock`.
    """
    try:
      self._autoscaling_status_lock.acquire()
      if not isinstance(status, AutoscalingStatus):
        raise ValueError("'%s' is not an AutoscalingStatus" % status)
      if self._autoscaling_status != status:
        log_fn("[Autoscaling] Changing status from %s to %s" % (self._autoscaling_status, status))
      self._autoscaling_status = status
    finally:
      self._autoscaling_status_lock.release()

  def before_run(self, run_context):
    if self.server is None and "server" in dir(self.estimator):
      self.server = self.estimator.server
      log_fn("Server is ready: %s" % self.server.target)
    log_fn("Before run")

