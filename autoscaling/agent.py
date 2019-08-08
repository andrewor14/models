#!/usr/bin/env python3

import errno
import os
import json
import socket
import threading
import time
import traceback

import tensorflow as tf
from tensorflow.python.distribute import cross_device_utils, distribute_coordinator
from tensorflow.python.eager import context

from autoscaling.client import connect, convert_port, AutoscalingClient
from autoscaling.service import listen_for_requests
from autoscaling.params import *
from deploy import cuda_helper

AUTOSCALING_MPI_COMMUNICATOR = None


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

  def __init__(self, num_gpus_per_worker=0):
    self.saved_variables = None
    # A lambda that returns a 2-tuple of
    #   (1) Number of batches processed in this epoch so far, and
    #   (2) Number of epochs processed so far.
    self.get_progress_method = None
    self.num_gpus_per_worker = num_gpus_per_worker

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
    if os.getenv("DATASET", "") == "cifar10":
      self.sync_interval_steps = 1

    # ========= Horovod stuff ==========

    # The MPI communicator that Horovod will use, if any
    self.mpi_communicator = None
    self.mpi_communicator_ranks = []

    # All host ports, indexed by the process rank
    # This is useful for expanding our communicator with processes in our world
    self.all_host_ports = []

    # A list of MPI intercommunicators created from spawning worker processes
    # These communicators will be merged into `self.mpi_communicator` on restart
    self.mpi_spawned_communicators = []

    if os.getenv("USE_HOROVOD", "").lower() == "true":
      from mpi4py import MPI
      self.all_host_ports = MPI.COMM_WORLD.allgather(self.host_port)
      self.mpi_communicator = MPI.COMM_SELF
      self.mpi_communicator_ranks = [MPI.COMM_WORLD.rank]
      global AUTOSCALING_MPI_COMMUNICATOR
      AUTOSCALING_MPI_COMMUNICATOR = self.mpi_communicator

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
    first_worker = os.getenv(AUTOSCALING_MASTER_HOST_PORT)
    if first_worker is None:
      first_worker = tf_config["cluster"]["worker"][0]
      first_worker = convert_port(first_worker)
    self.client = AutoscalingClient(first_worker)

    # If we are not part of the original cluster, request to join it
    # This fails if the master server is initializing, in which case we keep retrying
    if AUTOSCALING_MASTER_HOST_PORT in os.environ:
      log_fn("Joining cluster as %s" % self.host_port)
      while not self.client.master_server.join_cluster(self.host_port):
        log_fn("Master server is not ready to service our join request, trying again later.")
        time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
      log_fn("Master server accepted our join request")

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

  def using_horovod(self):
    """
    Return whether we use horovod to average gradients during training.
    """
    return self.mpi_communicator is not None

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
    # Our master host port may have changed, so we need to update our client
    new_master_host_port = convert_port(self.cluster_spec["worker"][0])
    self.client.reset(new_master_host_port)
    self.status = AutoscalingStatus.READY_TO_SYNC
    self.status_barrier(AutoscalingStatus.READY_TO_SYNC)
    self.sync_cluster_spec()
    # Set TF_CONFIG using the synced cluster spec
    new_tf_config = {"cluster": self.cluster_spec,\
      "task": {"type": self.task_type, "index": self.task_index}}
    new_tf_config = json.dumps(new_tf_config)
    log_fn("Setting TF_CONFIG = %s" % new_tf_config)
    os.environ["TF_CONFIG"] = new_tf_config
    # Update CUDA_VISIBLE_DEVICES with respect to new TF_CONFIG
    if self.num_gpus_per_worker > 0:
      cuda_helper.set_cuda_visible_devices(self.num_gpus_per_worker)
    # When using horovod, update our MPI communicator based on the new cluster spec
    if self.using_horovod():
      self.update_mpi_communicator_from_cluster_spec()
    self.status = AutoscalingStatus.RUNNING
    self.status_barrier(AutoscalingStatus.RUNNING)

  # TODO: move all MPI logic to a different file
  def add_workers_to_mpi_communicator(self, num_workers=1):
    """
    Prepare to add workers to our MPI communicator, which will be updated after a restart.
    This should only be called on the master server.
    """
    if not is_running(self.status):
      log_fn("Not adding workers because we are initializing")
      return False

    # Make sure adding the new workers does not exceed our world capacity
    spare_capacity = len(self.all_host_ports) - self.mpi_communicator.size
    if num_workers > spare_capacity:
      log_fn("Warning: unable to add %s workers because our current communicator "
        "has size %s and the world only has size %s. Adding %s workers instead." %\
        (num_workers, self.mpi_communicator.size, len(self.all_host_ports), spare_capacity))
      num_workers = spare_capacity

    # Find workers that are not already in our current cluster spec
    i = self.mpi_communicator.size
    current_workers = self.cluster_spec["worker"].copy()
    new_workers = []
    while len(new_workers) < num_workers:
      candidate = self.all_host_ports[i % len(self.all_host_ports)]
      if candidate not in current_workers and candidate not in new_workers:
        new_workers.append(candidate)
      i += 1
      if i > 2 * len(self.all_host_ports):
        # should never happen
        raise ValueError("Unable to find new workers to add to our MPI communicator")

    log_fn("ADDING THESE WORKERS %s" % new_workers)

    # Tell everyone in the new cluster spec to add the other workers
    # Note: we cannot use `self.client` here because it is not aware of the new workers yet
    all_workers = current_workers + new_workers
    for worker in all_workers:
      server = connect(convert_port(worker))
      server.add_workers(all_workers)
    return True

  def remove_workers_from_mpi_communicator(self, host_ports):
    """
    Prepare to remove workers from our MPI communicator, which will be updated after a restart.
    This should only be called on the master server.
    """
    if not is_running(self.status):
      log_fn("Not removing workers %s because we are initializing" % host_ports)
      return False

    log_fn("REMOVING THESE WORKERS %s" % host_ports)

    # Filter out workers that are not in our cluster spec
    removed_workers = []
    for host_port in host_ports:
      if host_port == self.host_port:
        log_fn("Warning: not removing worker %s because it is ourselves" % host_port)
      elif host_port not in self.cluster_spec["worker"]:
        log_fn("Warning: not removing worker %s because it is not in our cluster spec" % host_port)
      else:
        removed_workers.append(host_port)

    # Tell everyone involved to restart
    for host_port, server in self.client._servers.items():
      if host_port in removed_workers:
        # For removed workers, just tell them to run by themselves again
        server.set_pending_cluster_spec({"worker": [host_port]})
      else:
        server.remove_workers(removed_workers)
    return True

  def reset_mpi_communicator(self):
    """
    Remove everyone but ourselves from our MPI communicator.
    """
    workers_to_remove = self.cluster_spec["worker"].copy()
    workers_to_remove.remove(self.host_port)
    return self.remove_workers_from_mpi_communicator(workers_to_remove)

  def update_mpi_communicator_from_cluster_spec(self):
    """
    Expand our existing communicator by incorporating processes from our cluster spec.
    We assume all processes in the cluster spec are in the same MPI.COMM_WORLD as us.
    """
    from mpi4py import MPI
    if self.cluster_spec["worker"] == self.all_host_ports:
      self.mpi_communicator = MPI.COMM_WORLD.Dup()
      log_fn("Updated MPI communicator (size %s) to the original world" %\
        self.mpi_communicator.size)
    else:
      ranks = []
      for worker in self.cluster_spec["worker"]:
        ranks.append(self.all_host_ports.index(worker))
      self.mpi_communicator_ranks = ranks
      new_group = MPI.COMM_WORLD.group.Incl(ranks)
      self.mpi_communicator = MPI.COMM_WORLD.Create_group(new_group)
      log_fn("Updated MPI communicator (size %s) to match cluster spec %s" %\
        (self.mpi_communicator.size, self.cluster_spec))
    global AUTOSCALING_MPI_COMMUNICATOR
    AUTOSCALING_MPI_COMMUNICATOR = self.mpi_communicator


  # TODO: remove this
  def expand_mpi_communicator_with_spawned_workers(self):
    """
    Merge newly spawned workers, if any, into our existing communicator.
    """
    from mpi4py import MPI
    from deploy import mpi_helper
    # First, figure out our role
    comm = self.mpi_communicator
    is_joining = comm.rank == 0 and AUTOSCALING_MASTER_HOST_PORT in os.environ
    is_root = comm.rank == 0 and not is_joining
    # If we're joining, then expand once first
    if is_joining:
      comm = mpi_helper.expand(comm, MPI.Comm.Get_parent())
    # Note: MPI only allows us to expand once at a time, so we will do so in a loop.
    # In each iteration, the root will broadcast whether there are still spawned
    # communicators waiting to join. If so, all members of the existing communicator
    # will participate in a round of expansion.
    while True:
      should_expand = len(self.mpi_spawned_communicators) > 0 if is_root else False
      should_expand = comm.bcast(should_expand, root=0)
      if not should_expand:
        break
      if is_root:
        spawned_comm = self.mpi_spawned_communicators.pop(0)
        comm = mpi_helper.expand(comm, spawned_comm)
      else:
        comm = mpi_helper.expand(comm)
    self.mpi_communicator = comm

  # TODO: remove this
  def mpi_spawn_worker(self):
    """
    Spawn a worker process through MPI, succeeds only if called while running.
    The spawned worker, if any, is added to `self.mpi_spawned_communicators`.
    Return whether a worker was successfully spawned.
    """
    from deploy import mpi_helper
    if not self.using_horovod():
      raise ValueError("Spawn worker is only allowed when running with Horovod")
    if self.mpi_communicator.rank > 0:
      raise ValueError("Only the root can spawn workers")
    if not is_running(self.status):
      log_fn("Not spawning worker because we are initializing")
      return False
    new_worker_rank = self.mpi_communicator.size + len(self.mpi_spawned_communicators)
    env = {AUTOSCALING_MASTER_HOST_PORT: self.client.master_host_port}
    spawned_communicator = mpi_helper.spawn(new_worker_rank, env=env)
    self.mpi_spawned_communicators.append(spawned_communicator)
    log_fn("Spawned new worker through MPI")
    return True

  def on_restart(self):
    """
    Reset internal state in tensorflow on restart.
    """
    # Much of the internal tensorflow state needs to be cleared only when we are using a
    # distribution strategy. When using horovod, there is no strategy and so there is no need
    # to run the following.
    if not self.using_horovod():
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
        for server in self.client.servers:
          their_cluster_spec = json.dumps(server.get_cluster_spec())
          if my_cluster_spec != their_cluster_spec:
            failure_message = "... cluster spec sync failed (mine = %s, theirs = %s)" % (my_cluster_spec, their_cluster_spec)
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
      servers = self.client.servers
      statuses = [AutoscalingStatus(server.get_status()) for server in servers]
      if all([status in acceptable_statuses for status in statuses]):
        log_fn("... barrier reached! %s" % format_statuses(statuses))
        return statuses
      log_fn("... barrier not reached: %s" % format_statuses(statuses))
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)

  def save_variables(self, variables, session=None):
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
      self.saved_variables[v.name] = values[i]
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

    Return whether this process is restarting or terminating.
    """
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

