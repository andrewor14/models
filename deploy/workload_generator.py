#!/usr/bin/env python3

import os
import socket
import subprocess

EXECUTABLE = "bash"
RUN_SCRIPT = "run_distributed.sh"

# TODO: support other models
def get_synthetic_workload():
  return [
    ResNetWorkload("cifar10", 8, 64, num_steps=100),
    ResNetWorkload("cifar10", 4, 32, num_steps=8),
    ResNetWorkload("cifar10", 2, 16, num_steps=4)
  ]

class Job:
  """
  Helper class to represent jobs.
  """

  def __init__(self, job_id, workload, priority=0):
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
      num_gpus,
      batch_size,
      num_steps,
      num_epochs):
    self.num_gpus = num_gpus
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
    env["NUM_GPUS"] = "1"
    env["BATCH_SIZE"] = str(self.batch_size)
    if self.num_steps is not None:
      env["NUM_STEPS"] = str(self.num_steps)
    if self.num_epochs is not None:
      env["NUM_EPOCHS"] = str(self.num_epochs)
    env["ENABLE_ELASTICITY"] = "true"
    env ["MPI_VERBOSE"] = "true"
    print("Running workload %s with env %s" % (self, env))
    env.update(os.environ)
    return subprocess.Popen([EXECUTABLE, RUN_SCRIPT],\
      env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class ResNetWorkload(Workload):

  def __init__(
      self,
      dataset,
      num_gpus,
      batch_size,
      num_steps=None,
      num_epochs=None):
    super(ResNetWorkload, self).__init__(
      num_gpus, batch_size, num_steps, num_epochs)
    self.dataset = dataset

  def run(self, job_id, master_host, all_hosts, num_assigned_gpus, num_gpus_per_node, env={}):
    env = env.copy()
    env["MODEL"] = "resnet"
    env["DATASET"] = self.dataset
    return super().run(job_id, master_host, all_hosts, num_assigned_gpus, num_gpus_per_node, env)

  def __str__(self):
    return "ResNetWorkload(dataset=%s, num_gpus=%s, batch_size=%s, num_steps=%s, num_epochs=%s)" %\
      (self.dataset, self.num_gpus, self.batch_size, self.num_steps, self.num_epochs)

