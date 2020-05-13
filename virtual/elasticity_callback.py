#!/usr/bin/env python3

import copy
import time
import threading
import xmlrpc.server

from absl import logging
from mpi4py import MPI
import tensorflow as tf

from virtual import mpi_helper
from virtual.virtual_helper import *


class ElasticityCallback(tf.keras.callbacks.Callback):
  """
  A callback that maintains elasticity state for this process.
  """
  def __init__(self):
    tf_config = get_tf_config()
    task_type = tf_config["task"]["type"]
    task_index = tf_config["task"]["index"]
    self.cluster_spec = tf_config["cluster"]
    self.host_port = self.cluster_spec[task_type][task_index]

    # The cluster spec to apply next time we reinitialize, only exists on the master
    self.pending_cluster_spec = None

    # ========= MPI stuff ==========

    # The MPI communicator that Horovod will use
    self.mpi_communicator = MPI.COMM_WORLD.Dup()

    # A lock for controlling spawn variables
    self.spawn_lock = threading.Lock()

    # A map from rank to MPI intercommunicators created from spawning worker processes
    # These communicators will be merged into `self.mpi_communicator` on restart
    # Accesses must be guarded by `self.spawn_lock`
    self.spawned_communicators = {}

    # A queue of lists of ranks representing spawned workers to wait for.
    # Every call to `self.initialize` dequeues the first list and waits for all the
    # ranks in this list to join. This queue is populated by `self.spawn_workers`.
    # Accesses must be guarded by `self.spawn_lock`
    self.spawned_ranks_to_wait_for = []

    # The unique rank assigned to the next spawned worker, incremented every time a
    # worker is spawned. Only used on the master server.
    # Accesses must be guarded by `self.spawn_lock`.
    self.next_spawn_rank = len(self.cluster_spec["worker"])

    # ========= RPC server stuff ==========

    # RPC server's bound host port, only exists on the master
    self.rpc_server_host_port = None

    # If we are the master, listen for elasticity requests
    # If we are joining the cluster, send a join request to the master
    self.is_joining = ELASTICITY_MASTER_HOST_PORT in os.environ
    self.is_master = self.mpi_communicator.rank == 0 and not self.is_joining
    if self.is_master:
      self._start_server()
    elif self.is_joining:
      master_server_host_port = os.environ[ELASTICITY_MASTER_HOST_PORT]
      server_proxy = xmlrpc.client.ServerProxy("http://%s" % master_server_host_port)
      server_proxy.handle_join_request(self.host_port)
      self.maybe_expand_communicator()
      self.sync_cluster_spec()
      self.is_joining = False

  def _start_server(self):
    """
    Start a server that responds to elasticity requests.
    """
    host = self.host_port.split(":")[0]
    port = int(self.host_port.split(":")[1]) + ELASTICITY_PORT_OFFSET
    self.rpc_server_host_port = "%s:%s" % (host, port)
    logging.info("Listening for elasticity requests on %s" % self.rpc_server_host_port)
    server = xmlrpc.server.SimpleXMLRPCServer((host, port), logRequests=False, allow_none=True)
    server.register_function(self.get_cluster_spec)
    server.register_function(self.spawn_workers)
    server.register_function(self.handle_join_request)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()

  def sync_cluster_spec(self):
    """
    Synchronize cluster spec with everyone in our communicator.
    """
    logging.info("Attempting to sync cluster spec with everyone")
    self.mpi_communicator.barrier()
    while True:
      my_cluster_spec = json.dumps(self.cluster_spec)
      all_cluster_specs = self.mpi_communicator.allgather(json.dumps(self.cluster_spec))
      if all([cs == my_cluster_spec for cs in all_cluster_specs]):
        break
      logging.info("... not everyone is synced yet")
      time.sleep(RETRY_INTERVAL_SECONDS)
      self.cluster_spec = self.mpi_communicator.bcast(self.cluster_spec, root=0)
    logging.info("... cluster spec synced: %s" % self.cluster_spec)
    self.mpi_communicator.barrier()

  def get_cluster_spec(self):
    return copy.deepcopy(self.cluster_spec)

  def handle_join_request(self, host_port):
    if not self.is_master:
      raise ValueError("Only the master can handle join requests")
    pending_cluster_spec = self.get_cluster_spec()
    pending_cluster_spec["worker"].append(host_port)
    self.pending_cluster_spec = pending_cluster_spec

  def spawn_workers(self, num_additional_workers):
    """
    Spawn one or more worker processes through MPI.
    The spawned workers, if any, are added to `self.spawned_communicators`.
    This succeeds only if called on the root while the root is running batches.
    """
    if not self.is_master:
      raise ValueError("Only the master can spawn workers")
    if num_additional_workers <= 0:
      raise ValueError("Number of workers to spawn must be a positive number")
    spawn_ranks = []
    logging.info("Spawning %s worker(s) through MPI" % num_additional_workers)
    with self.spawn_lock:
      starting_rank = self.next_spawn_rank
      spawn_ranks.extend(list(range(starting_rank, starting_rank + num_additional_workers)))
      self.spawned_ranks_to_wait_for.append(spawn_ranks)
      self.next_spawn_rank += num_additional_workers
    # Set up environment for new processes
    def do_spawn(new_worker_rank):
      env = {ELASTICITY_MASTER_HOST_PORT: self.rpc_server_host_port}
      spawned_communicator = mpi_helper.spawn_process(new_worker_rank, env=env)
      with self.spawn_lock:
        self.spawned_communicators[new_worker_rank] = spawned_communicator
    # Spawn processes in parallel; each call to MPI_SPAWN takes a few seconds
    spawn_threads = []
    for r in spawn_ranks:
      t = threading.Thread(target=do_spawn, args=[r])
      t.start()
      spawn_threads.append(t)
    for tt in spawn_threads:
      tt.join()
    logging.info("Done spawning")

  def maybe_expand_communicator(self, spawned_ranks):
    """
    Merge newly spawned workers, if any, into our existing communicator.

    This method must be called at the same time on all the processes participating
    in the final communicator. The process is driven by the root, where `spawned_ranks`
    specifies the specific communicators to merge into the existing one. This value
    is not read on other processes. Must be called while holding `self.spawn_lock`.
    """
    # First, figure out our role
    comm = self.mpi_communicator
    # If we're joining, then expand once first
    if self.is_joining:
      comm = mpi_helper.expand_communicator(comm, MPI.Comm.Get_parent())
    # Note: MPI only allows us to expand once at a time, so we will do so in a loop.
    # In each iteration, the root will broadcast whether there are still spawned
    # communicators waiting to join. If so, all members of the existing communicator
    # will participate in a round of expansion.
    while True:
      should_expand = len(spawned_ranks) > 0 if self.is_master else False
      should_expand = comm.bcast(should_expand, root=0)
      if not should_expand:
        break
      if self.is_master:
        spawned_rank = spawned_ranks.pop(0)
        if spawned_rank not in self.spawned_communicators:
          raise ValueError("Attempted to expand with unknown rank %s" % spawned_rank)
        spawned_comm = self.spawned_communicators[spawned_rank]
        comm = mpi_helper.expand_communicator(comm, spawned_comm)
      else:
        comm = mpi_helper.expand_communicator(comm)
    self.mpi_communicator = comm

  def on_batch_end(self, batch, logs=None):
    """
    Adjust cluster configuration if necessary.
    """
    # If we are the master, check whether we should restart
    should_restart = False
    if self.is_master:
      if self.pending_cluster_spec is not None:
        num_workers_expected = 0
        num_workers_joined =\
          len(set(self.pending_cluster_spec["worker"]) - set(self.cluster_spec["worker"]))
        with self.spawn_lock:
          if len(self.spawned_ranks_to_wait_for) > 0:
            num_workers_expected = len(self.spawned_ranks_to_wait_for[0])
        should_restart = num_workers_joined >= num_workers_expected
    should_restart = self.mpi_communicator.bcast(should_restart, root=0)
    # On restart, first expand our communicator, then use the new
    # communicator to sync cluster specs with everyone else
    # TODO: joined workers should also fetch trained variables
    # TODO: transfer virtual nodes to new workers
    if should_restart:
      with self.spawn_lock:
        new_ranks = []
        if len(self.spawned_ranks_to_wait_for) > 0:
          new_ranks = self.spawned_ranks_to_wait_for.pop(0)
        self.maybe_expand_communicator(new_ranks)
      if self.is_master:
        self.cluster_spec = self.pending_cluster_spec
        self.pending_cluster_spec = None
      self.sync_cluster_spec()

