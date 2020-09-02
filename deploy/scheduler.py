#!/usr/bin/env python3

from collections import OrderedDict
import copy
import socket
import subprocess
import sys
import threading
import xmlrpc.server

SCHEDULER_PORT=18383

def get_scheduler_client(host):
  """
  Return a client that can communicate with the scheduler server.
  """
  return xmlrpc.client.ServerProxy("http://%s:%s" % (host, SCHEDULER_PORT))

class WorkloadScheduler:

  def __init__(self, all_hosts, num_gpus_per_node):
    self.next_job_id = 0
    self.num_gpus_per_node = num_gpus_per_node
    self.assign_lock = threading.Lock()

    # Mapping from host to a list of `num_gpus_per_node` items, where each
    # entry represents a GPU and contains the job ID the GPU is assigned to,
    # or None if the GPU is unassigned.
    # All accesses must be guarded by `self.assign_lock`.
    self.gpu_assignment = {}
    with self.assign_lock:
      for host in all_hosts:
        self.gpu_assignment[host] = [None] * num_gpus_per_node

    # Mapping from job ID to the elasticity master host for that job,
    # used to issue resize requests
    self.elasticity_master_hosts = {}

    # Listen for requests to get the GPUs assigned to a given job
    server = xmlrpc.server.SimpleXMLRPCServer(
      (socket.gethostname(), SCHEDULER_PORT), logRequests=False, allow_none=True)
    server.register_function(self.get_assigned_gpus)
    server.register_function(self.report_released_gpus)
    t = threading.Thread(target=server.serve_forever)
    t.start()
    print("Listening for scheduler requests on port %s" % SCHEDULER_PORT)

    # TODO: delete
    from deploy.workload_generator import ResNetWorkload
    self.run_workload(ResNetWorkload("cifar10", 8, 32, num_steps=3))

  def get_assigned_gpus(self, job_id=None):
    """
    Return the set of GPUs assigned to the given job, in the same format as `self.gpu_assignment`.
    An entry of -1 means the GPU is not assigned to this job.
    If `job_id` is None, simply return a copy of `self.gpu_assignment`.
    """
    from virtual.elasticity_callback import GPU_BLACKLIST_VALUE
    with self.assign_lock:
      assigned_gpus = copy.deepcopy(self.gpu_assignment)
    if job_id is not None:
      for host in assigned_gpus.keys():
        for i, j in enumerate(assigned_gpus[host]):
          if j != job_id:
            assigned_gpus[host][i] = GPU_BLACKLIST_VALUE
    return assigned_gpus

  def report_released_gpus(self, job_id, released_gpus):
    """
    Called by a job to report the GPUs it released after a transition.
    `released_gpus` is a list of 2-tuple (host, gpu_index).
    """
    with self.assign_lock:
      for host, gpu_index in released_gpus:
        if self.gpu_assignment[host][gpu_index] != job_id:
          raise ValueError("Job %s released GPU %s on host %s but that GPU was never assigned to that job" %\
            (job_id, gpu_index, host))
        self.gpu_assignment[host][gpu_index] = None

  def run_workload(self, workload):
    from virtual.elasticity_callback import assign_gpus
    with self.assign_lock:
      assigned_gpus = assign_gpus(workload.num_gpus, self.gpu_assignment)
      # TODO: what to do if num GPUs assigned is not a power of 2?
      if len(assigned_gpus) == 0:
        print("Warning: scheduler was unable to find any GPUs to run workload %s" % workload)
        return
      job_id = self.next_job_id
      self.next_job_id += 1
      # Update GPU assignment
      for host, gpu_index in assigned_gpus:
        self.gpu_assignment[host][gpu_index] = job_id
      # Use the host containing the first GPU as the master host
      # Put this host at the front of the host list to ensure MPI launches a process there first
      master_host = assigned_gpus[0][0]
      hosts = list(self.gpu_assignment.keys()).copy()
      hosts.remove(master_host)
      hosts.insert(0, master_host)
      self.elasticity_master_hosts[job_id] = master_host
    # Run the job in the background
    # On exit, release all GPUs assigned to this job
    def wait_for_process(job_id, hosts, num_gpus_per_node):
      print("Job %s started" % job_id)
      workload.run(job_id, hosts, num_gpus_per_node).wait()
      print("Job %s finished" % job_id)
      with self.assign_lock:
        for host in self.gpu_assignment.keys():
          for i, j in enumerate(self.gpu_assignment[host]):
            if j == job_id:
              self.gpu_assignment[host][i] = None
        del self.elasticity_master_hosts[job_id]
      # TODO: trigger schedule
    t = threading.Thread(target=wait_for_process, args=(job_id, hosts, self.num_gpus_per_node,))
    t.setDaemon(True)
    t.start()

def main():
  args = sys.argv
  if len(args) != 3:
    print("Usage: ./scheduler.py [host1,host2,host3...] [num_gpus_per_node]")
    sys.exit(1)
  all_hosts = args[1]
  num_gpus_per_node = int(args[2])
  all_hosts = all_hosts.split(",")
  WorkloadScheduler(all_hosts, num_gpus_per_node)

if __name__ == "__main__":
  main()

