#!/usr/bin/env python3

import copy
import math
import os
import random
import socket
import subprocess
import sys
import time
import threading
import xmlrpc.server

from deploy.scheduler_event import *
from deploy.scheduler_workload import *

SCHEDULER_PORT = 18383
LOOP_INTERVAL_SECONDS = 1
DEBUG = os.getenv("DEBUG", "").lower() == "true"

def get_scheduler_client(host):
  """
  Return a client that can communicate with the scheduler server.
  """
  return xmlrpc.client.ServerProxy("http://%s:%s" % (host, SCHEDULER_PORT))

class WorkloadScheduler:

  def __init__(self, all_hosts, num_gpus_per_node):
    self.next_job_id = 0
    self.num_gpus_per_node = num_gpus_per_node

    # A global lock for all operations accessing our data structures
    self.lock = threading.RLock()

    # Mapping from host to a list of `num_gpus_per_node` items, where each
    # entry represents a GPU and contains the job ID the GPU is assigned to,
    # or None if the GPU is unassigned.
    self.gpu_assignment = {}
    with self.lock:
      for host in all_hosts:
        self.gpu_assignment[host] = [None] * num_gpus_per_node

    # A list of Workload objects waiting to be scheduled
    self.job_queue = []

    # A list of Event objects waiting to be processed
    self.event_queue = []

    # A list of Event objects that were processed but failed
    # These events will be placed back into the event queue in the next round
    self.failed_events = []

    # A list of jobs currently running on the cluster
    self.running_jobs = []

    # Mapping from job ID to the elasticity master host for that job,
    # used to issue resize requests
    self.elasticity_master_hosts = {}

    # Listen for requests to get the GPUs assigned to a given job
    server = xmlrpc.server.SimpleXMLRPCServer(
      (socket.gethostname(), SCHEDULER_PORT), logRequests=False, allow_none=True)
    server.register_function(self.submit_job)
    server.register_function(self.get_assigned_gpus)
    server.register_function(self.report_released_gpus)
    t = threading.Thread(target=server.serve_forever)
    t.start()
    print("Listening for scheduler requests on port %s" % SCHEDULER_PORT)

    # Start event processing loop
    # If an event was not processed successfully, try it again after a short delay
    def event_loop(scheduler):
      while True:
        with scheduler.lock:
          scheduler.event_queue.extend(self.failed_events)
          self.failed_events = []
          while len(scheduler.event_queue) > 0:
            e = scheduler.event_queue.pop(0)
            print("* Processing event %s" % e)
            if not e.process(scheduler):
              self.failed_events.append(e)
        time.sleep(LOOP_INTERVAL_SECONDS)
    t = threading.Thread(target=event_loop, args=(self,))
    t.setDaemon(True)
    t.start()
    print("Started event loop...")

  def submit_job(self, gpu_demand, priority, num_steps=100):
    """
    Submit a job to the scheduler queue.
    TODO: support different workloads
    """
    with self.lock:
      job_id = self.next_job_id
      self.next_job_id += 1
      job = Job(job_id, ResNetWorkload("cifar10", gpu_demand, 32, num_steps), priority)
      self.job_queue.append(job)
      self.job_queue.sort(key=lambda j: -j.priority)
      self.event_queue.append(ScheduleEvent())

  def schedule(self):
    """
    Main scheduling loop triggered by ScheduleEvents.

    This triggers the scheduling of new jobs (RunJobEvent) and the resizing of
    existing jobs (ResizeEvent) to match the weighted fair sharing allocations.

    If an existing `RunJobEvent` or `ResizeEvent` is already in the queue, it means
    there are jobs waiting for resources to be released by other jobs in the system.
    In this case, we put another ScheduleEvent back into the queue and try again later
    so as to avoid excessive cluster churn.
    """
    with self.lock:
      # If there is an ongoing resize or run job event, try again later
      for e in self.event_queue:
        if isinstance(e, ResizeEvent) or isinstance(e, RunJobEvent):
          return False

      current_allocations = self.get_current_allocations()
      fair_allocations = self.get_weighted_fair_shares(self.running_jobs)

      if DEBUG:
        print("schedule: current_allocations %s" % current_allocations)

      # Keep scheduling new jobs until existing higher priority jobs are affected
      new_jobs = []
      while len(self.job_queue) > 0:
        candidate_job = self.job_queue[0]
        potential_allocations = self.get_weighted_fair_shares(
          self.running_jobs + new_jobs + [candidate_job])
        if DEBUG:
          print("schedule: potential_allocations %s" % potential_allocations)
        # Run this job only if doing so would not affect the allocations of higher
        # priority jobs that are already running
        run_new_job = True
        for job in self.running_jobs:
          if job.priority > candidate_job.priority and\
              potential_allocations[job.job_id] < fair_allocations[job.job_id]:
            run_new_job = False
        if run_new_job:
          fair_allocations = potential_allocations
          new_jobs.append(self.job_queue.pop(0))
        else:
          break

      if DEBUG:
        print("schedule: fair_allocations %s" % fair_allocations)

      # Resize existing jobs and/or run new jobs to match the fair allocation
      if current_allocations != fair_allocations:
        assert(len(fair_allocations) >= len(current_allocations))
        # Resize old jobs if the allocation has changed
        for job_id in current_allocations.keys():
          if current_allocations[job_id] != fair_allocations[job_id]:
            self.event_queue.append(ResizeEvent(job_id, fair_allocations[job_id]))
        # Schedule all new jobs
        for job in new_jobs:
          self.event_queue.append(RunJobEvent(job, fair_allocations[job.job_id]))

    return True

  def get_current_allocations(self):
    """
    Return a mapping from job ID to the number of GPUs allocated to the job.
    """
    with self.lock:
      current_allocations = {}
      for gpus in self.gpu_assignment.values():
        for j in gpus:
          if j is not None:
            if j not in current_allocations:
              current_allocations[j] = 0
            current_allocations[j] += 1
    return current_allocations

  def get_current_allocation(self, job_id):
    """
    Return the number of GPUs allocated to a job.
    """
    return self.get_current_allocations()[job_id]

  def get_weighted_fair_shares(self, jobs):
    """
    Get the weighted fair shares for the given jobs.

    One modification to the standard weighted fair sharing algorithm here is we require
    all GPU allocations are powers of 2. This makes it easier to resize the cluster while
    ensuring the amount of data processed on each GPU per batch stays a power of 2.

    However, this additional requirement means the number of GPUs allocated may be less
    than the total number of GPUs that are available in the cluster. We try to mitigate
    this by attempting to double the allocation for some jobs to fill the void.

    Return a dictionary mapping job ID to the number of GPUs allocated to the job.
    """
    with self.lock:
      # Mapping of job ID to num GPUs allocated
      gpu_allocations = {}
      unallocated_jobs = jobs.copy()
      num_unallocated_gpus = sum([len(gpus) for gpus in self.gpu_assignment.values()])
      if len(jobs) > num_unallocated_gpus:
        raise ValueError("Unable to satisfy all %s jobs with %s GPUs" %\
          (len(jobs), num_unallocated_gpus))

      # Compute weighted fair shares
      while len(unallocated_jobs) > 0:
        # Mapping of job ID to the job's weighted fair share in this round
        weighted_fair_shares = {}
        priority_sum = sum([j.priority for j in unallocated_jobs])
        for job in unallocated_jobs:
          wfs = num_unallocated_gpus * job.priority / priority_sum
          if wfs < 1:
            wfs = 1
          # Round down to the nearest power of 2
          wfs = 2 ** math.floor(math.log2(wfs))
          weighted_fair_shares[job.job_id] = wfs

        # Note: The sum of all weighted fair shares could exceed the number of unallocated
        # GPUs if, e.g. there are many jobs and one has very high priority and demand.
        # If this is the case, we solve this by halving the fair shares of a subset of
        # jobs in increasing priority order.
        i = 0
        sorted_jobs = sorted(unallocated_jobs, key=lambda j: j.priority)
        while sum(weighted_fair_shares.values()) > num_unallocated_gpus\
            and i < len(sorted_jobs):
          job_id = sorted_jobs[i].job_id
          if weighted_fair_shares[job_id] > 1:
            weighted_fair_shares[job_id] /= 2
          else:
            i += 1
        assert(sum(weighted_fair_shares.values()) <= num_unallocated_gpus)

        # Make sure all fair shares are integers
        for jid, wfs in weighted_fair_shares.items():
          if not float(wfs).is_integer():
            print("Warning: Weighted fair share for job %s was not an integer (%s), rounding" %\
              (jid, wfs))
          weighted_fair_shares[jid] = int(wfs)

        # Allocate GPUs to jobs whose fair share exceed their demands
        # This frees up some GPUs to be distributed among the remaining jobs next round
        jobs_allocated_this_round = []
        for job in unallocated_jobs:
          gpu_demand = job.workload.gpu_demand
          if weighted_fair_shares[job.job_id] >= gpu_demand:
            gpu_allocations[job.job_id] = gpu_demand
            num_unallocated_gpus -= gpu_demand
            jobs_allocated_this_round.append(job)

        # If we allocated GPUs this round, remove those jobs from consideration next round
        if len(jobs_allocated_this_round) > 0:
          for job in jobs_allocated_this_round:
            unallocated_jobs.remove(job)
        else:
          # No job's fair share exceed its demand, just allocate based on the fair shares
          for job in unallocated_jobs:
            wfs = weighted_fair_shares[job.job_id]
            gpu_allocations[job.job_id] = wfs
            num_unallocated_gpus -= wfs
          unallocated_jobs = []

      # There may still be unallocated GPUs because we round down to the nearest power of 2.
      # Here we try fill those GPUs by attempting to double the allocations of a subset of
      # jobs in descending priority order.
      if num_unallocated_gpus > 0:
        for job in sorted(jobs, key=lambda j: -j.priority):
          current_allocation = gpu_allocations[job.job_id]
          # Ensure doubling does not exceed the job's GPU demand
          while num_unallocated_gpus >= current_allocation and\
              job.workload.gpu_demand >= 2 * current_allocation:
            num_unallocated_gpus -= current_allocation
            current_allocation *= 2
          gpu_allocations[job.job_id] = current_allocation
      assert(num_unallocated_gpus >= 0)

      return gpu_allocations

  def get_assigned_gpus(self, job_id=None):
    """
    Return the set of GPUs assigned to the given job, in the same format as `self.gpu_assignment`.
    An entry of -1 means the GPU is not assigned to this job.
    If `job_id` is None, simply return a copy of `self.gpu_assignment`.
    """
    from virtual.elasticity_callback import GPU_BLACKLIST_VALUE
    with self.lock:
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
    with self.lock:
      for host, gpu_index in released_gpus:
        if self.gpu_assignment[host][gpu_index] != job_id:
          raise ValueError("Job %s released GPU %s on host %s but that GPU was never assigned to that job" %\
            (job_id, gpu_index, host))
        self.gpu_assignment[host][gpu_index] = None

  def get_available_gpus(self, n):
    """
    Return a set of GPUs not currently assigned to any job, in the format of a list of
    2-tuples (host, gpu_index).
    """
    from virtual.elasticity_callback import assign_gpus
    return assign_gpus(n, self.gpu_assignment)

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

