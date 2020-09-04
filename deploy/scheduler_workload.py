#!/usr/bin/env python3

import json
import os
import socket
import subprocess

EXECUTABLE = "bash"
RUN_SCRIPT = "run_distributed.sh"

def workload_from_name(name):
  """
  Return a workload class from the given identifier.
  """
  if name.lower() == "resnet":
    return ResNetWorkload
  else:
    raise ValueError("Unknown workload name: %s" % name)

def generate_schedule_from_trace(path):
  """
  Construct a list of 2-tuples (arrival_time, job) from a JSON trace.
  `arrival_time` refers to seconds elapsed after the scheduler started.
  """
  schedule = []
  with open(path) as f:
    for i, j in enumerate(json.load(f)):
      arrival_time = j["arrival_time"]
      workload_cls = workload_from_name(j["workload"])
      workload = workload_cls.from_json(j)
      priority = j.get("priority") or 1
      job = Job(i, workload, priority)
      schedule.append((arrival_time, job))
  return schedule

class Job:
  """
  Helper class to represent jobs.
  """

  def __init__(self, job_id, workload, priority=1):
    self.job_id = job_id
    self.workload = workload
    self.priority = priority

  def __str__(self):
    return "Job(%s, %s, %s)" % (self.job_id, self.workload, self.priority)

class Workload:
  """
  Abstract base class for all workloads.
  """

  def __init__(
      self,
      gpu_demand,
      batch_size,
      num_steps,
      num_epochs):
    self.gpu_demand = gpu_demand
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.num_epochs = num_epochs

  def run(self, job_id, master_host, all_hosts, num_assigned_gpus, num_gpus_per_node, env={}):
    """
    Run this workload asynchronously and return the process object.
    """
    env = env.copy()
    env["JOB_ID"] = str(job_id)
    env["SCHEDULER_ADDR"] = socket.gethostname()
    env["MASTER_HOST"] = master_host
    env["MPI_HOSTS"] = ",".join(all_hosts)
    env["NUM_GPUS_PER_NODE"] = str(num_gpus_per_node)
    env["NUM_NODES"] = str(num_assigned_gpus)
    # Due to contention in the cluster, we may not be granted our full GPU demand
    # In this case, our demand should be a multiple of the number of GPUs assigned,
    # and this multiple is the number of virtual nodes per device
    if self.gpu_demand % num_assigned_gpus != 0:
      raise ValueError("GPU demand (%s) was not a multiple of number of " % gpu_demand +
        "GPUs assigned (%s) for job %s" % (num_assigned_gpus, job_id))
    env["NUM_VIRTUAL_NODES_PER_DEVICE"] = str(int(self.gpu_demand / num_assigned_gpus))
    env["NUM_GPUS"] = "1"
    env["BATCH_SIZE"] = str(self.batch_size)
    if self.num_steps is not None:
      env["NUM_STEPS"] = str(self.num_steps)
    if self.num_epochs is not None:
      env["NUM_EPOCHS"] = str(self.num_epochs)
    env["ENABLE_ELASTICITY"] = "true"
    env["RUN_TAG"] = "job%s" % job_id
    env.update(os.environ)
    return subprocess.Popen([EXECUTABLE, RUN_SCRIPT],\
      env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class ResNetWorkload(Workload):

  def __init__(
      self,
      dataset,
      gpu_demand,
      batch_size,
      num_steps=None,
      num_epochs=None):
    super(ResNetWorkload, self).__init__(gpu_demand, batch_size, num_steps, num_epochs)
    self.dataset = dataset

  def run(self, job_id, master_host, all_hosts, num_assigned_gpus, num_gpus_per_node, env={}):
    env = env.copy()
    env["MODEL"] = "resnet"
    env["DATASET"] = self.dataset
    return super().run(job_id, master_host, all_hosts, num_assigned_gpus, num_gpus_per_node, env)

  def __str__(self):
    return "ResNetWorkload(dataset=%s, gpu_demand=%s, batch_size=%s, num_steps=%s, num_epochs=%s)" %\
      (self.dataset, self.gpu_demand, self.batch_size, self.num_steps, self.num_epochs)

  @staticmethod
  def from_json(j):
    return ResNetWorkload(j["dataset"], j["gpu_demand"], j["batch_size"],
      j.get("num_steps"), j.get("num_epochs"))

