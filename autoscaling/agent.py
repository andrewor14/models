#!/usr/bin/env python3

import errno
import io
import json
import os
import socket
import threading
import time
import traceback
import xmlrpc.client

from mpi4py import MPI
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import cross_device_utils, distribute_coordinator
from tensorflow.python.eager import context

from autoscaling import autoscaling_helper
from autoscaling.client import convert_port, AutoscalingClient
from autoscaling.service import listen_for_requests
from autoscaling.params import *
from deploy import cuda_helper, mpi_helper


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

  def __init__(self, num_gpus_per_worker=0, global_batch_size=0, use_horovod=False):
    self.saved_variables = None
    # A lambda that returns a 3-tuple:
    #  (1) Number of batches processed in this epoch so far,
    #  (2) Number of epochs processed so far, and
    #  (3) Number of batches per epoch
    self.get_progress_method = None
    self.bootstrap_progress_method = None
    self.checkpoint_restart_num_workers = None
    self.num_gpus_per_worker = num_gpus_per_worker
    self.global_batch_size = global_batch_size
    self.use_horovod = use_horovod

    # Parse this process' host port from TF_CONFIG
    tf_config = get_tf_config()
    self.task_type = tf_config["task"]["type"]
    self.task_index = tf_config["task"]["index"]
    self.cluster_spec = tf_config["cluster"]
    self.host_port = self.cluster_spec[self.task_type][self.task_index]
    self.host_port = find_available_port(self.host_port)

    # Syncing at the end of each step adds some overhead
    # If we are running a tiny dataset, we may want to sync less often
    self.step_count = 0
    self.sync_interval_steps = int(os.getenv(AUTOSCALING_SYNC_INTERVAL_STEPS, "1"))

    # ========= MPI stuff ==========

    # The MPI communicator that Horovod will use
    self.mpi_communicator = MPI.COMM_WORLD.Dup()

    # A lock for controlling spawn variables
    # This must be acquired BEFORE `self.pending_cluster_spec_lock` to avoid deadlocks
    self.spawn_lock = threading.Lock()

    # A list of MPI intercommunicators created from spawning worker processes
    # These communicators will be merged into `self.mpi_communicator` on restart
    # Accesses must be guarded by `self.spawn_lock`
    self.mpi_spawned_communicators = []

    # The unique rank assigned to the next spawned worker, incremented every time a
    # worker is spawned. Only set on the master server.
    # Accesses must be guarded by `self.spawn_lock`.
    self.mpi_next_rank = None

    # A queue of the number of spawned workers to wait for on restart
    # The first item represents the number of workers to wait for the next
    # time `self.initialize` is called. A new item is appended every time
    # `self.mpi_spawn_workers` succeeds.
    # Accesses must be guarded by `self.spawn_lock`
    self.num_spawned_workers_to_wait_for = []

    # ========= Autoscaling stuff ==========

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

    # Start autoscaling server
    listen_for_requests(self, convert_port(self.host_port))

    # Start autoscaling client, connected to the autoscaling server on the first worker
    first_worker = tf_config["cluster"]["worker"][0]
    first_worker = convert_port(first_worker)
    self.client = AutoscalingClient(first_worker)

    # Spawned workers join after running one batch by themselves
    # This avoids blocking existing workers while the spawned workers are starting up
    self.joined = AUTOSCALING_MASTER_HOST_PORT not in os.environ

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

  def initialize(self):
    """
    Ensure everyone sees the same cluster spec, then set TF_CONFIG accordingly.
    This should be called on start up and after every time we restart.
    """
    log_fn("Initializing")

    # Wait for all the workers spawned in the last round to join
    num_workers_joined = 0
    num_workers_to_wait_for = 0
    with self.spawn_lock:
      if len(self.num_spawned_workers_to_wait_for) > 0:
        num_workers_to_wait_for = self.num_spawned_workers_to_wait_for[0]
        log_fn("Waiting for %s spawned worker(s) to join" % num_workers_to_wait_for)
        while num_workers_joined < num_workers_to_wait_for:
          with self.pending_cluster_spec_lock:
            if self.pending_cluster_spec is not None:
              num_workers_joined =\
                len(set(self.pending_cluster_spec["worker"]) - set(self.cluster_spec["worker"]))
          time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
    self.status = AutoscalingStatus.READY_TO_SYNC
    self.status_barrier(AutoscalingStatus.READY_TO_SYNC)

    # Check if cluster membership changed. If so, update cluster spec accordingly.
    with self.spawn_lock:
      with self.pending_cluster_spec_lock:
        if self.pending_cluster_spec is not None:
          num_workers_joined =\
            len(set(self.pending_cluster_spec["worker"]) - set(self.cluster_spec["worker"]))
          log_fn("%s spawned worker(s) have joined" % num_workers_joined)
          # Update the number of workers we need to wait for next time
          # Note: we can't just pop the first item of `self.num_spawned_workers_to_wait_for`
          # because more workers may have joined in addition to the ones we were waiting for.
          if len(self.num_spawned_workers_to_wait_for) > 0:
            num_workers_to_pop = num_workers_joined
            while num_workers_to_pop > 0:
              if num_workers_to_pop >= self.num_spawned_workers_to_wait_for[0]:
                num_workers_to_pop -= self.num_spawned_workers_to_wait_for.pop(0)
              else:
                self.num_spawned_workers_to_wait_for[0] -= num_workers_to_pop
                num_workers_to_pop = 0
          # Apply pending cluster spec
          self.apply_cluster_spec(self.pending_cluster_spec)
          self.pending_cluster_spec = None
    self.sync_cluster_spec()
    self.task_index = self.cluster_spec["worker"].index(self.host_port)

    # Set TF_CONFIG using the synced cluster spec
    new_tf_config = {"cluster": self.cluster_spec,\
      "task": {"type": self.task_type, "index": self.task_index}}
    new_tf_config = json.dumps(new_tf_config)
    log_fn("Setting TF_CONFIG = %s" % new_tf_config)
    os.environ["TF_CONFIG"] = new_tf_config
    # Update CUDA_VISIBLE_DEVICES with respect to new TF_CONFIG
    if self.num_gpus_per_worker > 0:
      cuda_helper.set_cuda_visible_devices(self.num_gpus_per_worker)
    # When using horovod, delete TF_CONFIG to avoid interference from tensorflow
    if self.use_horovod:
      del os.environ["TF_CONFIG"]
    # Check if we need to expand our communicator
    if self.joined:
      with self.spawn_lock:
        self.maybe_expand_mpi_communicator(num_workers_joined)
    # Tell tensorflow our batch size has changed
    autoscaling_helper.LOCAL_BATCH_SIZE =\
      self.global_batch_size // len(self.cluster_spec["worker"])
    log_fn("Local batch size = %s" % autoscaling_helper.LOCAL_BATCH_SIZE)
    self.status = AutoscalingStatus.RUNNING
    self.status_barrier(AutoscalingStatus.RUNNING)

  def maybe_expand_mpi_communicator(self, n_times):
    """
    Merge newly spawned workers, if any, into our existing communicator.

    This method must be called at the same time on all the processes participating
    in the final communicator. The process is driven by the root, where `n_times`
    specifies the number of times to expand the communicator. This value is not
    read on other processes.

    Must be called while holding `self.spawn_lock`.
    """
    # First, figure out our role
    comm = self.mpi_communicator
    is_joining = comm.rank == 0 and AUTOSCALING_MASTER_HOST_PORT in os.environ
    is_root = comm.rank == 0 and not is_joining
    # If we're the root, make sure we can expand the requested number of times
    if is_root and n_times > len(self.mpi_spawned_communicators):
      raise ValueError("Cannot expand %s times when we only spawned %s workers" %\
        (n_times, len(self.mpi_spawned_communicators)))
    # If we're joining, then expand once first
    if is_joining:
      comm = mpi_helper.expand(comm, MPI.Comm.Get_parent())
    # Note: MPI only allows us to expand once at a time, so we will do so in a loop.
    # In each iteration, the root will broadcast whether there are still spawned
    # communicators waiting to join. If so, all members of the existing communicator
    # will participate in a round of expansion.
    while True:
      should_expand = n_times > 0 if is_root else False
      should_expand = comm.bcast(should_expand, root=0)
      if not should_expand:
        break
      if is_root:
        spawned_comm = self.mpi_spawned_communicators.pop(0)
        comm = mpi_helper.expand(comm, spawned_comm)
      else:
        comm = mpi_helper.expand(comm)
      n_times -= 1
    self.mpi_communicator = comm

  def mpi_spawn_workers(self, num_workers):
    """
    Spawn one or more worker processes through MPI.

    The spawned workers, if any, are added to `self.mpi_spawned_communicators`.
    This succeeds only if called on the root while the root is running batches.
    Return whether spawn was attempted.
    """
    if self.mpi_communicator.rank > 0:
      raise ValueError("Only the root can spawn workers")
    if not is_running(self.status):
      log_fn("Not spawning worker because we are initializing")
      return False
    log_fn("Spawning %s worker(s) through MPI" % num_workers)
    with self.spawn_lock:
      if self.mpi_next_rank is None:
        self.mpi_next_rank = len(self.cluster_spec["worker"])
      starting_rank = self.mpi_next_rank
      self.mpi_next_rank += num_workers
    def do_spawn(new_worker_rank):
      env = {AUTOSCALING_MASTER_HOST_PORT: self.client.master_host_port}
      spawned_communicator = mpi_helper.spawn(new_worker_rank, env=env)
      with self.spawn_lock:
        self.mpi_spawned_communicators.append(spawned_communicator)
        self.num_spawned_workers_to_wait_for.append(num_workers)
    for i in range(num_workers):
      threading.Thread(target=do_spawn, args=[starting_rank + i]).start()
    return True

  def join_cluster(self):
    """
    If we are not part of the original cluster, request to join it.
    This fails if the master server is initializing, in which case we keep retrying.
    """
    if self.joined:
      raise ValueError("Already joined cluster!")
    master_host_port = os.getenv(AUTOSCALING_MASTER_HOST_PORT)
    if master_host_port is None:
      raise ValueError("AUTOSCALING_MASTER_HOST_PORT not set on spawned worker")
    # Our client currently thinks we are the master, so we need to reset it
    self.client.reset(master_host_port)
    # Wait until master is ready to accept our join request
    log_fn("Joining cluster as %s" % self.host_port)
    while not self.client.master_server_rpc(lambda s: s.join_cluster(self.host_port)):
      log_fn("Master server is not ready to service our join request, trying again later.")
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
    log_fn("Master server accepted our join request")
    # Fetch progress from the master so we can start on the same step as everyone else
    # Note: we must wait until after the master is READY_TO_SYNC, otherwise we may get the
    # wrong progress
    self.status = AutoscalingStatus.READY_TO_SYNC
    self.status_barrier(AutoscalingStatus.READY_TO_SYNC)
    (num_batches_processed_this_epoch, num_epochs_processed, num_batches_per_epoch) =\
      self.bootstrap_progress_method()
    self.step_count = num_epochs_processed * num_batches_per_epoch +\
      num_batches_processed_this_epoch
    # Fetch model parameters from existing workers
    # TODO: fetch from everyone, not just the master
    log_fn("Fetching model parameters from existing workers")
    fetch_start = time.time()
    self.saved_variables = self.client.master_server_rpc(lambda s: s.get_saved_variables())
    log_fn("Fetched %s model parameters, took %s seconds" %\
      (len(self.saved_variables), time.time() - fetch_start))
    # Tell tensorflow which step and epoch to restart from
    autoscaling_helper.STEP_NUMBER = num_batches_processed_this_epoch
    autoscaling_helper.EPOCH_NUMBER = num_epochs_processed
    self.joined = True

  def on_restart(self):
    """
    Reset internal state in tensorflow on restart.
    """
    # Much of the internal tensorflow state needs to be cleared only when we are using a
    # distribution strategy. When using horovod, there is no strategy and so there is no need
    # to run the following.
    if not self.use_horovod:
      log_fn("Resetting internal tensorflow state")
      # Note: tensorflow maintains a thread local variable to keep track of the existing server
      # If such a server exists, then tensorflow will simply reuse it. Here we clear this variable
      # to avoid this behavior, because we *do* want it to start a new server with a different
      # server def.
      distribute_coordinator._thread_local.__dict__.clear()
      # Note: tensorflow maintains a thread local variable to keep track of collective ops.
      # If we don't clear this, then tensorflow will reuse the old op, which has a wrong number
      # of workers, and hang without any error messages.
      cross_device_utils._thread_local.__dict__.clear()
      # Destroy the existing graph used internally by keras, otherwise adding workers hangs
      # when calling the batch normalization layer. This is caused by a mismatch in the instance
      # keys used in collective ops between old and new workers in the cluster. Resetting the
      # global variables used by keras solves this problem.
      tf.keras.backend.clear_session()
      # Finally, reset all other internal state stored in the context
      context.context().reset()

  def sync_cluster_spec(self):
    """
    Retry until all the following are true
      (1) Our cluster spec contains our host name
      (2) We have the same cluster spec as everyone else
      (3) Everyone is SYNCED

    Our status must be READY_TO_SYNC before we call this method, and RUNNING when we return.
    """
    log_fn("Attempting to sync cluster spec with everyone")
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
        cluster_specs = self.client.all_servers_rpc(lambda s: json.dumps(s.get_cluster_spec()))
        for cluster_spec in cluster_specs:
          if cluster_spec != my_cluster_spec:
            failure_message = "... cluster spec sync failed"
      # If no failure so far, then we are synced, so we should transition to SYNCED
      if not failure_message:
        log_fn("... cluster spec synced: %s" % my_cluster_spec)
        self.status = AutoscalingStatus.SYNCED
        self.status_barrier(AutoscalingStatus.SYNCED)
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
      statuses = self.client.all_servers_rpc(lambda s: AutoscalingStatus(s.get_status()))
      if all([status in acceptable_statuses for status in statuses]):
        log_fn("... barrier reached! %s" % format_statuses(statuses))
        return statuses
      log_fn("... barrier not reached: %s" % format_statuses(statuses))
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)

  def save_variables(self, variables, session=None, compress=False):
    """
    Save the values of the given variables to memory.

    If `session` is provided, use it to compute the value of the variables.
    Otherwise, we assume this is running in eager mode, in which case we can
    just directly access the values of the variables.
    """
    save_start = time.time()
    if session is not None:
      session.graph._finalized = False
      values = session.run(variables)
    else:
      # We're running in eager mode
      values = [v.value() for v in variables]
    save_end = time.time()
    self.saved_variables = {}
    for i, v in enumerate(variables):
      value = values[i]
      if compress:
        buf = io.BytesIO()
        np.savez_compressed(buf, value=value)
        value = buf.getvalue()
      self.saved_variables[v.name] = value
    log_fn("Saved %s variables in memory, took %s seconds" %\
      (len(self.saved_variables), save_end - save_start))

  def restore_variables(self, variables, session=None):
    """
    Restore the values of saved variables from memory, if any.
    This assumes `saved_variables` is not None.

    If `session` is provided, use it to restore the values of the variables.
    Otherwise, we assume this is running in eager mode, in which case we can
    just directly update the variables.
    """
    try:
      if len(self.saved_variables) != len(variables):
        raise ValueError("Number of saved variables (%s) != number of variables provided (%s)" %\
          (len(self.saved_variables), len(variables)))
      log_fn("Restoring %s variables from memory" % len(variables))
      restore_ops = []
      restore_start = time.time()
      for var in variables:
        val = self.saved_variables[var.name]
        # Value may be compressed, in which case it will be in bytes
        if isinstance(val, xmlrpc.client.Binary):
          val = val.data
        if isinstance(val, bytes):
          data = np.load(io.BytesIO(val))
          if len(data.files) != 1:
            raise ValueError("Unknown compression format")
          val = data[data.files[0]]
        update_fn = lambda var, val: var.assign(val)
        restore_ops.append(
          tf.distribute.get_strategy().extended.update(var, update_fn, args=(val,)))
      if session is not None:
        session.graph._finalized = False
        session.run(restore_ops)
      restore_end = time.time()
      log_fn("Restored %s variables from memory, took %s seconds" %\
        (len(variables), restore_end - restore_start))
    finally:
      self.saved_variables = None

  def step_end(self):
    """
    Listen for changes in cluster membership and react by restarting the server.

    This method is called at the end of each step. Each process undergoes three
    state changes:
      (1) From RUNNING to [NOT_]PENDING_RESTART, which specifies whether
          this process has observed the cluster membership change, and
      (2) From that to [NOT_]READY_TO_RESTART, which specifies whether
          this process has observed that *everyone* is pending restart.
      (3) From that to either RUNNING, RESTARTING, or TERMINATED

    Each of these state changes are followed by a synchronization barrier to
    ensure that the processes either all restart or keep running. The only
    exception is TERMINATED, which happens when a process realizes that he is
    actually not in the new cluster configuration. When this happens, the caller
    of this method should exit the training loop for this process.

    Return whether this process should reinitialize. New workers should
    reinitialize after the first step and existing workers should reinitialize
    if there are new workers and everyone is aware of their existence.
    """
    # If this is a spawned worker, join the cluster after the first step
    if not self.joined:
      self.join_cluster()
      return True
    # Only sync every N steps
    self.step_count += 1
    if self.step_count % self.sync_interval_steps != 0:
      return False
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
    # If even one worker is not ready to restart, then we just keep running
    if AutoscalingStatus.NOT_READY_TO_RESTART in statuses:
      self.status = AutoscalingStatus.RUNNING
      self.status_barrier(AutoscalingStatus.RUNNING, quiet=True)
      return False
    else:
      # If the new cluster spec does not contain us, then that means we are removed
      should_terminate = False
      with self.pending_cluster_spec_lock:
        should_terminate = self.host_port not in self.pending_cluster_spec["worker"]
      if should_terminate:
        log_fn("Received signal to terminate")
        self.status = AutoscalingStatus.TERMINATED
      else:
        log_fn("Received signal to restart server")
        self.status = AutoscalingStatus.RESTARTING
        self.status_barrier(AutoscalingStatus.RESTARTING)
      return True

# ================== HELPER METHODS ==================

def log_fn(msg):
  tf.logging.info("[Autoscaling agent] %s" % msg)

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

def find_available_port(host_port, max_retries=10):
  """
  Find smallest port larger than or equal to `base_port` that is not in use.
  Return new host port with a potentially modified port.
  """
  split = host_port.split(":")
  host = split[0]
  base_port = int(split[1])
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    offset = 0
    while offset <= max_retries:
      target_port = base_port + offset
      try:
        sock.bind(("localhost", target_port))
        return "%s:%s" % (host, target_port)
      except socket.error as e:
        if e.errno == errno.EADDRINUSE:
          log_fn("Warning: Attempted to bind to port %s, but it was already in use" %\
            target_port)
      offset += 1
    raise Exception("All ports in range [%s-%s] are already in use!" %\
      (base_port, base_port + max_retries))
  finally:
    sock.close()

