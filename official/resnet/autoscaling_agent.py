#!/usr/bin/env python3

import os
import json
import threading
import time
import traceback

import tensorflow as tf
from tensorflow.python.distribute import distribute_coordinator
from tensorflow.python.distribute import cross_device_utils

from official.resnet.autoscaling_client import convert_port, AutoscalingClient
from official.resnet.autoscaling_service import listen_for_autoscaling_requests
from official.resnet.autoscaling_params import *


class AutoscalingAgent:
  """
  An agent that keeps track of autoscaling state for this process.

  This agent communicates with other agents on remote processes to
  ensure that the autoscaling status on this process transitions in
  synchrony with everyone else.

  Upon detecting a change in cluster membership, the agents vote to
  decide when to restart. Restarting requires a unanimous consensus
  among all the agents.
  """

  def __init__(self):
    self.saved_variables = None
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
      self._status = s

  def set_global_batch_size(self, initial_local_batch_size):
    """
    Set the global batch size, which is fixed throughout training.

    We first try to get the global batch size from the master autoscaling server.
    If the master doesn't have it yet, e.g. because we're bootstrapping the initial set
    of workers, then we simply set it ourselves based on our cluster spec.

    This must be called before the first step.
    """
    if self.global_batch_size is None:
      self.global_batch_size = self.client.master_server.get_global_batch_size()
    if self.global_batch_size is None:
      num_workers = len(self.cluster_spec["worker"])
      self.global_batch_size = initial_local_batch_size * num_workers

  @property
  def local_batch_size(self):
    """
    Return the local (per worker) batch size.
    This changes every time the number of workers changes.
    """
    if self.global_batch_size is None:
      raise ValueError("Global batch size must be set before accessing local batch size")
    return int(self.global_batch_size * 1.0 / len(self.cluster_spec["worker"]))

  def initialize(self):
    """
    Ensure everyone sees the same cluster spec, then set TF_CONFIG accordingly.
    This should be called on start up and after every time we restart.
    """
    log_fn("Initializing")
    # Check if cluster membership changed. If so, update cluster spec accordingly.
    with self.pending_cluster_spec_lock:
      if self.pending_cluster_spec is not None:
        self.apply_cluster_spec(self.pending_cluster_spec)
        self.pending_cluster_spec = None
    self.status = AutoscalingStatus.READY_TO_SYNC
    self.sync_cluster_spec()
    # TODO: do this through set_server_def instead (currently hangs)
    # from tensorflow.python.eager import context
    # from tensorflow.python.training.server_lib import _make_server_def
    # server_def = _make_server_def(
    #   self.pending_cluster_spec, self.task_type, self.task_index, "grpc", None)
    # context.context().set_server_def(server_def)
    new_tf_config = {"cluster": self.cluster_spec,\
      "task": {"type": self.task_type, "index": self.task_index}}
    os.environ["TF_CONFIG"] = json.dumps(new_tf_config)
    log_fn("Setting TF_CONFIG = %s" % new_tf_config)
    # Note: tensorflow maintains a thread local variable to keep track of the existing server
    # If such a server exists, then tensorflow will simply reuse it. Here we clear this variable
    # to avoid this behavior, because we *do* want it to start a new server with a different
    # server def.
    distribute_coordinator._thread_local.__dict__.clear()
    # Note: tensorflow maintains a thread local variable to keep track of collective ops.
    # If we don't clear this, then tensorflow will reuse the old op, which has a wrong number
    # of workers, and hang without any error messages.
    cross_device_utils._thread_local.__dict__.clear()

  def sync_cluster_spec(self):
    """
    Retry until all the following are true
      (1) Our cluster spec contains our host name
      (2) We have the same cluster spec as everyone else
      (3) Everyone is SYNCED

    Our status must be READY_TO_SYNC before we call this method, and RUNNING when we return.
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
        self.status = AutoscalingStatus.RUNNING
        self.status_barrier(AutoscalingStatus.RUNNING)
        return
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

  def status_barrier(self, target, quiet=False):
    """
    Wait until everyone has reached the target autoscaling status(es).

    The given target can be one status or a list of statuses.
    This requires the caller to already be in one of the target statuses.
    Return the list of everyone's status once the target is reached.
    """
    targets = [target] if not isinstance(target, list) else target
    log_fn = lambda _: None if quiet else log_fn
    # Check if we have reached the target ourselves
    my_status = self.status
    if my_status not in targets:
      raise ValueError("Current autoscaling status %s must match barrier target(s): %s" %\
        (my_status, targets))
    # A process may have passed the same barrier already, in which case it will be in
    # one of the next statuses and that's OK. Terminated is also accepted.
    acceptable_statuses = targets + [AutoscalingStatus.TERMINATED]
    for t in targets:
      acceptable_statuses.extend(get_next_statuses(t))
    log_fn("Waiting for everyone to reach %s" % format_statuses(targets))
    while True:
      servers = self.client.servers
      statuses = [AutoscalingStatus(server.get_status()) for server in servers]
      if all([status in acceptable_statuses for status in statuses]):
        log_fn("... barrier reached! %s" % format_statuses(statuses))
        return statuses
      log_fn("... barrier not reached: %s" % format_statuses(statuses))
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)

  def get_trainable_variables(self):
    """
    Return a list of trainable variables.
    """
    return tf.global_variables()

  def save_variables(self, sess):
    """
    Save the values of savable variables to memory.
    """
    trainable_variables = self.get_trainable_variables()
    save_start = time.time()
    values = sess.run(trainable_variables)
    save_end = time.time()
    self.saved_variables = {}
    for i, v in enumerate(trainable_variables):
      self.saved_variables[v.name] = values[i]
    log_fn("Saved %s variables in memory, took %s seconds" %\
      (len(self.saved_variables), save_end - save_start))

  def restore_variables(self, sess):
    """
    Restore the values of saved variables from memory, if any.
    This assumes `saved_variables` is not None.
    """
    try:
      trainable_variables = self.get_trainable_variables()
      if len(self.saved_variables) != len(trainable_variables):
        raise ValueError("Number of saved variables (%s) differ from number of trainable variables (%s)" %\
          (len(self.saved_variables), len(trainable_variables)))
      log_fn("Restoring %s variables from memory" % len(trainable_variables))
      restore_ops = []
      restore_start = time.time()
      for var in trainable_variables:
        val = self.saved_variables[var.name]
        update_fn = lambda var, val: var.assign(val)
        restore_ops.append(tf.distribute.get_strategy().extended.update(var, update_fn, args=(val,)))
      sess.run(restore_ops)
      restore_end = time.time()
      log_fn("Restored %s variables from memory, took %s seconds" %\
        (len(trainable_variables), restore_end - restore_start))
    finally:
      self.saved_variables = None

  def step_end(self):
    """
    Listen for changes in cluster membership and react by restarting the server.

    This method is called at the end of each step. Each process undergoes two
    state changes:
      (1) From RUNNING to [NOT_]PENDING_RESTART, which specifies whether
          this process has observed the cluster membership change, and
      (2) From that to [NOT_]READY_TO_RESTART, which specifies whether
          this process has observed that *everyone* is pending restart.

    Each of these state changes are followed by a synchronization barrier to
    ensure that the processes either all restart or keep running.

    There is a third potential state change, TERMINATED, which happens when
    a process realizes that he is actually not in the new cluster configuration.
    When this happens, the caller of this method should exit the training loop
    for this process.

    Return whether this process is restarting.
    """
    # Check if cluster membership has changed
    with self.pending_cluster_spec_lock:
      if self.pending_cluster_spec is not None:
        self.status = AutoscalingStatus.PENDING_RESTART
      else:
        self.status = AutoscalingStatus.NOT_PENDING_RESTART
    # Check if everyone has seen the cluster membership change
    statuses = self.status_barrier(\
      [AutoscalingStatus.PENDING_RESTART, AutoscalingStatus.NOT_PENDING_RESTART],
      quiet=True)
    if AutoscalingStatus.NOT_PENDING_RESTART in statuses:
      self.status = AutoscalingStatus.NOT_READY_TO_RESTART
    else:
      self.status = AutoscalingStatus.READY_TO_RESTART
    # Check if everyone is ready to restart
    statuses = self.status_barrier(\
      [AutoscalingStatus.READY_TO_RESTART, AutoscalingStatus.NOT_READY_TO_RESTART],
      quiet=True)
    if AutoscalingStatus.NOT_READY_TO_RESTART in statuses:
      self.status = AutoscalingStatus.RUNNING
    else:
      # Do restart
      if self.host_port not in self.pending_cluster_spec["worker"]:
        log_fn("Received signal to terminate")
        self.status = AutoscalingStatus.TERMINATED
      else:
        log_fn("Received signal to restart server")
      return True


# ================== HELPER METHODS ==================

def log_fn(msg):
  msg = "[Autoscaling agent] %s" % msg
  tf.compat.v1.logging.info(msg)

def get_tf_config():
  """
  Return TF_CONFIG as a dictionary, assuming it was set.
  """
  tf_config = os.getenv("TF_CONFIG")
  if tf_config is None:
    raise ValueError("'TF_CONFIG' must be set.")
  return json.loads(tf_config.replace("'", "\""))

def log_exceptions(fn):
  """
  Do not swallow exceptions.
  """
  try:
    fn()
  except Exception as e:
    log_fn("%s (%s)" % (e, e.__class__.__name__))
    traceback.print_exc()
    raise e

