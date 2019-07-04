#!/usr/bin/env python3

import os
import json
import threading
import time

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.training.server_lib import _make_server_def

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
    # Accesses must be guarded by `self._status_lock`
    self._status = AutoscalingStatus.READY_TO_SYNC
    self._status_lock = threading.Lock()

    # The cluster spec to apply next time we reinitialize
    # Note: We can't apply the new cluster spec right away because doing so will
    # signal to other workers that they can go ahead and build their graphs, which
    # might be inconsistent with ours if we haven't finished reinitializing yet.
    # Accesses must be guarded by `self.pending_cluster_spec_lock`.
    self.pending_cluster_spec = None
    self.pending_cluster_spec_lock = threading.Lock()    

    # Parse this process' host port from TF_CONFIG
    tf_config = get_tf_config()
    self.task_type = tf_config["task"]["type"]
    self.task_index = tf_config["task"]["index"]
    self.cluster_spec = tf_config["cluster"]
    self.host_port = self.cluster_spec[self.task_type][self.task_index]

    # Start autoscaling server
    listen_for_autoscaling_requests(self, convert_port(self.host_port))

    # Start autoscaling client, connected to the autoscaling server on the first worker
    first_worker = os.getenv(AUTOSCALING_MASTER_HOST_PORT)
    if first_worker is None:
      first_worker = tf_config["cluster"]["worker"][0]
      first_worker = convert_port(first_worker)
    self.client = AutoscalingClient(first_worker)

    # Request to join the cluster
    self.client.master_server.join_cluster(self.host_port)

  @property
  def status(self):
    """
    Atomic getter for `self._status`, guarded by `self._status_lock`.
    """
    with self._status_lock:
      return self._status

  @status.setter
  def status(self, s):
    """
    Atomic setter for `self._status`, guarded by `self._status_lock`.
    """
    with self._status_lock:
      if not isinstance(s, AutoscalingStatus):
        raise ValueError("'%s' is not an AutoscalingStatus" % s)
      if self._status != s:
        log_fn("Changing status from %s to %s" % (self._status, s))
      self._status = s

  def sync_cluster_spec(self):
    """
    Retry until all the following are true
      (1) Our cluster spec contains our host name
      (2) We have the same cluster spec as everyone else
      (3) Everyone's status is SYNCED or SETTING_UP

    Our status must be READY_TO_SYNC before we call this method, and SETTING_UP when we return.

    Return the cluster spec that was synced in dictionary form.
    """
    log_fn("Attempting to sync cluster spec with everyone")
    self.status_barrier(AutoscalingStatus.READY_TO_SYNC)
    self.status = AutoscalingStatus.SYNCING
    self.status_barrier(AutoscalingStatus.SYNCING)
    while True:
      failure_message = None
      my_cluster_spec = json.dumps(self.client.cluster_spec)
      # (1) Does our cluster spec contain our host name?
      if self.host_port not in self.client.hosts:
        failure_message = "... cluster spec does not contain this host (%s)" % self.host_port
      # (2) Do we have the same cluster spec as everyone else?
      if not failure_message:
        for server in self.client.servers:
          their_cluster_spec = json.dumps(server.get_cluster_spec())
          if my_cluster_spec != their_cluster_spec:
            failure_message = "... cluster spec sync failed"
      # If no failure so far, then we are synced, so we should transition to SYNCED
      if not failure_message:
        log_fn("... cluster spec synced: %s" % my_cluster_spec)
        self.status = AutoscalingStatus.SYNCED
        self.status_barrier(AutoscalingStatus.SYNCED)
        self.status = AutoscalingStatus.SETTING_UP
        return json.loads(my_cluster_spec)
      # On failure, reset client with cluster spec from the master autoscaling server
      log_fn("%s, trying again in %s second(s)" %\
        (failure_message, AUTOSCALING_RETRY_INTERVAL_SECONDS))
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
      self.client.reset()
      self.apply_cluster_spec(self.client.cluster_spec)

  def apply_cluster_spec(self, cluster_spec):
    """
    Update our fields to match the provided cluster spec (python dict).
    """
    self.cluster_spec = cluster_spec
    tasks = cluster_spec[self.task_type]
    if self.host_port in tasks:
      self.task_index = tasks.index(self.host_port)
    ## Calculate new per device batch size from old global batch size
    ## If we're joining an existing cluster, just get it from the master
    #if self.global_batch_size is None:
    #  self.global_batch_size = self.client.master_server.get_global_batch_size()
    #if self.global_batch_size is not None:
    #  per_device_batch_size = int(self.global_batch_size * 1.0 / len(worker_hosts) / self.num_gpus)
    #  self.params = self.params._replace(batch_size=per_device_batch_size)

  def status_barrier(self, target, soft=False):
    """
    Wait until everyone has reached the target autoscaling status or the one after it.

    This requires the caller to already be in the target status.
    If `soft` is True then this returns after the first attempt.
    Return whether the target status barrier was reached by everyone.
    """
    if self.status != target:
      raise ValueError("Current autoscaling status %s must match barrier target %s" %\
        (self.status, target))
    log_fn("Waiting for everyone to reach %s" % target)
    target_next = get_next_status(target)
    while True:
      servers = self.client.servers
      statuses = [AutoscalingStatus(server.get_status()) for server in servers]
      statuses_str = [str(status).split(".")[-1] for status in statuses]
      # Don't wait for terminated processes
      if all([status == target or status == target_next or\
          status == AutoscalingStatus.TERMINATED for status in statuses]):
        log_fn("... barrier reached! %s" % statuses_str)
        return True
      log_fn("... barrier not reached: %s" % statuses_str)
      if soft:
        return False
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)

  def server_is_ready(self):
    """
    Return whether the server is ready.
    We get a pointer to the server through the estimator object.
    If the `server` field exists, then we consider the server to be ready.
    """
    if "server" not in dir(self.estimator):
      return False
    if self.server is None:
      self.server = self.estimator.server
      log_fn("Server is ready: %s" % self.server.target)
    return True

  def log_exception(self, closure):
    """
    Stop swallowing exceptions guys.
    """
    try:
      closure()
    except Exception as e:
      import traceback
      log_fn("ERROR: %s (%s)" % (e, e.__class__.__name__))
      traceback.print_exc()
      raise e

  def on_restart(self):
    """
    Check if cluster membership changed. If so, update server def accordingly.
    """
    log_fn("On restart")
    with self.pending_cluster_spec_lock:
      if self.pending_cluster_spec is not None:
        server_def = _make_server_def(
          self.pending_cluster_spec, self.task_type, self.task_index, "grpc", None)
        log_fn("Setting server def to %s" % server_def)
        context.context().set_server_def(server_def)
        log_fn("Server def was set!")
        self.apply_cluster_spec(self.pending_cluster_spec)
        self.pending_cluster_spec = None

  def do_begin(self):
    """
    Make sure everyone has the same cluster membership information, then transition to RUNNING.
    """
    log_fn("Begin")
    self.sync_cluster_spec()
    self.status_barrier(AutoscalingStatus.SETTING_UP)
    self.status = AutoscalingStatus.RUNNING

  def do_after_run(self, run_context, run_values):
    """
    Listen for changes in cluster membership and react by restarting the server.
    """
    log_fn("After run")
    if not self.server_is_ready():
      return
    # If there is a pending cluster spec, then it means cluster membership changed
    # When this happens, update our server def and request a restart
    with self.pending_cluster_spec_lock:
      if self.pending_cluster_spec is not None:
        # Make sure we're ready to sync
        if self.status != AutoscalingStatus.READY_TO_SYNC:
          log_fn("Changing cluster membership: %s" % self.pending_cluster_spec)
          self.status_barrier(AutoscalingStatus.RUNNING)
          self.status = AutoscalingStatus.READY_TO_SYNC
        # TODO: Add a synchronization barrier here to make sure everyone has had
        # the chance to transition to READY_TO_SYNC
        # Once everyone is READY_TO_SYNC, we can go ahead and restart
        # Note: we can't just block here because other processes may rely on us to make progress
        # Therefore, we need to keep running the benchmark until everyone is READY_TO_SYNC
        if self.status == AutoscalingStatus.READY_TO_SYNC:
          if self.status_barrier(AutoscalingStatus.READY_TO_SYNC, soft=True):
            # If we're not in the new cluster spec, then that means we were removed
            # Otherwise, we should save our variables and restart
            if self.host_port not in self.pending_cluster_spec["worker"]:
              log_fn("Received signal to terminate")
              self.status = AutoscalingStatus.TERMINATED
            else:
              log_fn("Received signal to restart server")
              # TODO: save variables if we're rebuilding the graph
            run_context.request_stop()
          else:
            log_fn("Not restarting because not everyone is READY_TO_SYNC yet")

  def begin(self):
    self.log_exception(self.do_begin)

  def after_run(self, run_context, run_values):
    self.log_exception(lambda: self.do_after_run(run_context, run_values))

