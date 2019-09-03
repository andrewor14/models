#!/usr/bin/env python3

import copy
import json
import time
import threading
import xmlrpc.server

import tensorflow as tf

from autoscaling.client import convert_port, connect
from autoscaling.params import *
from deploy import mpi_helper


def log_fn(msg):
  tf.logging.info("[Autoscaling service]: %s" % msg)

def listen_for_requests(agent, host_port):
  '''
  Start a server listening for autoscaling requests

  This is a simple RPC server that exposes an interface for adjusting
  the number of workers in a running job.
  '''
  log_fn("Listening for autoscaling requests on host port %s" % host_port)
  split = host_port.split(":")
  server = xmlrpc.server.SimpleXMLRPCServer(
    (split[0], int(split[1])), logRequests=False, allow_none=True)
  server.register_introspection_functions()
  server.register_multicall_functions()
  server.register_instance(AutoscalingService(agent))
  t = threading.Thread(target=server.serve_forever)
  t.daemon = True
  t.start()

class AutoscalingService:
  '''
  A service for handling autoscaling requests.
  '''
  def __init__(self, agent):
    self.agent = agent

  def get_status(self):
    return self.agent.status.value

  def get_cluster_spec(self):
    return copy.deepcopy(self.agent.cluster_spec)

  def get_progress(self):
    '''
    Return a 3-tuple:
      (1) Number of batches processed in this epoch so far,
      (2) Number of epochs processed so far, and
      (3) Number of batches per epoch
    '''
    if self.agent.get_progress_method is not None:
      return self.agent.get_progress_method()
    return (None, None, None)

  def get_saved_variables(self):
    '''
    Return a map of saved model variables indexed by variable name.
    This is used for bootstrapping new workers.
    '''
    return self.agent.saved_variables

  def request_attach(self):
    '''
    Request a previously removed worker to re-attach to the cluster.
    Return this process' rank when it was first spawned.
    '''
    log_fn("Received request to re-attach to cluster")
    self.agent.detached_mode = False
    return int(os.environ[mpi_helper.MPI_SPAWN_RANK])

  def assign_cuda_visible_devices(self, host_port):
    '''
    Assign CUDA_VISIBLE_DEVICES for the given host port on a first come first served basis.
    Return the CUDA_VISIBLE_DEVICES assigned.
    '''
    if self.agent.mpi_communicator.rank != 0 and self.agent.joined:
      raise ValueError("Received assign CUDA_VISIBLE_DEVICES request on non-master")
    # Place dummy values for existing hosts first
    device_map = self.agent.cuda_visible_devices_map
    if len(device_map) == 0:
      for worker in self.agent.cluster_spec["worker"]:
        device_map[worker] = []
    # Assign it
    if host_port not in device_map or len(device_map[host_port]) == 0:
      occupied_cuda_visible_devices = []
      for hp, dev in device_map.items():
        if hp.split(":")[0] == host_port.split(":")[0]:
          occupied_cuda_visible_devices.extend(dev)
      # Find a device that is not occupied
      # Keep around the original CUDA_VISIBLE_DEVICES because we will override it
      if ORIGINAL_CUDA_VISIBLE_DEVICES not in os.environ:
        os.environ[ORIGINAL_CUDA_VISIBLE_DEVICES] = os.environ[CUDA_VISIBLE_DEVICES]
      cuda_visible_devices = [int(d) for d in os.environ[ORIGINAL_CUDA_VISIBLE_DEVICES].split(",")]
      available_cuda_visible_devices = list(set(cuda_visible_devices) -\
        set(occupied_cuda_visible_devices))
      if len(available_cuda_visible_devices) < self.agent.num_gpus_per_worker:
        raise ValueError("Out of GPUs! Available: %s, requested: %s" %\
          (available_cuda_visible_devices, self.agent.num_gpus_per_worker))
      device_map[host_port] = available_cuda_visible_devices[:self.agent.num_gpus_per_worker]
    return device_map[host_port]

  def join_cluster(self, host_port, rank):
    '''
    Handle a join request, only called on the master server.

    The join request is rejected if it is received during initialization.
    Return whether the join request has been accepted.
    '''
    log_fn("Received join cluster request from %s" % host_port)
    # Prevent unexpected workers from interfering with cluster spec sync
    if is_syncing(self.agent.status):
      log_fn("Rejecting join request from %s because we are syncing cluster specs" % host_port)
      return False
    # Only let the specific ranks we are waiting for join our cluster
    # Note: we assume the only time consuming operations requiring `spawn_lock` occur when
    # the process is in one of the syncing statuses. This means it is OK for us to hold this
    # lock here without blocking for too long, since we already make sure that join requests
    # that happen during syncing won't even get to this point.
    with self.agent.spawn_lock:
      if len(self.agent.spawned_ranks_to_wait_for) > 0 and\
          rank not in self.agent.spawned_ranks_to_wait_for[0]:
        log_fn("Rejecting join request from %s (rank %s) because it is not one of %s" %\
          (host_port, rank, self.agent.spawned_ranks_to_wait_for[0]))
        return False
    cluster_spec = self.get_cluster_spec()
    ps_hosts = cluster_spec["ps"] if "ps" in cluster_spec else []
    worker_hosts = cluster_spec["worker"]
    hosts = ps_hosts + worker_hosts
    if host_port not in hosts:
      # Wait until client is ready
      while self.agent.client is None:
        log_fn("... autoscaling client is not ready yet, waiting %s second(s)" %\
          AUTOSCALING_RETRY_INTERVAL_SECONDS)
        time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
      # Tell everyone to add this worker.
      # Note: Calling client.add_worker directly will hang because this server
      # is single threaded and so cannot process the add_workers request asynchronously.
      # Thus, we need to treat the master server (ourselves) separately.
      self.agent.client.all_servers_rpc(lambda s: s.add_workers([host_port]), except_master=True)
      self.add_workers([host_port])
      # Note: There may be pending workers that are not part of the autoscaling client yet.
      # Here we manually tell them to add this new worker. In the future there may be a
      # cleaner way to do this.
      with self.agent.pending_cluster_spec_lock:
        if self.agent.pending_cluster_spec is not None:
          pending_workers = self.agent.pending_cluster_spec["worker"]
          pending_workers = list(set(pending_workers) - set(self.get_cluster_spec()["worker"]))
          if host_port in pending_workers:
            pending_workers.remove(host_port)
          for pending_worker in pending_workers:
            log_fn("Telling pending worker %s to add worker %s" % (pending_worker, host_port))
            self.agent.client.rpc_without_connection_problems(
              lambda: connect(convert_port(pending_worker)).add_workers([host_port]))
    else:
      log_fn("Warning: received join request from a worker who had already joined: %s" % host_port)
    return True

  def _get_or_create_pending_cluster_spec(self):
    '''
    Return the existing pending cluster spec or create a new one.
    The caller must hold `self.agent.pending_cluster_spec_lock`.
    '''
    if self.agent.pending_cluster_spec is None:
      self.agent.pending_cluster_spec = self.get_cluster_spec()
    return self.agent.pending_cluster_spec

  def add_workers(self, host_ports):
    log_fn("Handling add workers request: %s" % host_ports)
    with self.agent.pending_cluster_spec_lock:
      cluster_spec = self._get_or_create_pending_cluster_spec()
      cluster_spec["worker"].extend(host_ports)

  def remove_workers(self, host_ports):
    log_fn("Handling remove workers request: %s" % host_ports)
    with self.agent.pending_cluster_spec_lock:
      cluster_spec = self._get_or_create_pending_cluster_spec()
      for hp in host_ports:
        if hp in cluster_spec["worker"]:
          cluster_spec["worker"].remove(hp)
        else:
          log_fn("Warning: not removing unknown worker %s" % hp)

  def set_pending_cluster_spec(self, cluster_spec):
    log_fn("Handling set pending cluster spec request: %s" % cluster_spec)
    with self.agent.pending_cluster_spec_lock:
      self.agent.pending_cluster_spec = cluster_spec

  def spawn_workers(self, num_workers):
    log_fn("Handling spawn_worker request")
    return self.agent.mpi_spawn_workers(num_workers)

