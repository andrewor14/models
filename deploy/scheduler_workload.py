#!/usr/bin/env python3

import json
import math
import os
import random
import socket
import subprocess
import sys

EXECUTABLE = "bash"
RUN_SCRIPT = "run_distributed.sh"

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

  def __init__(self, gpu_demand, batch_size, num_steps=None, num_epochs=None):
    self.gpu_demand = gpu_demand
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.num_epochs = num_epochs

  def run(self, scheduler, job_id, master_host, num_assigned_gpus, env={}):
    """
    Run this workload asynchronously and return the process object.
    This assumes the caller holds `scheduler.lock`.
    """
    env = env.copy()
    env["JOB_ID"] = str(job_id)
    if scheduler.addr is not None:
      env["SCHEDULER_ADDR"] = scheduler.addr
    env["MASTER_HOST"] = master_host
    env["MPI_HOSTS"] = ",".join(list(scheduler.gpu_assignment.keys()).copy())
    env["NUM_GPUS_PER_NODE"] = str(scheduler.num_gpus_per_node)
    env["NUM_NODES"] = str(num_assigned_gpus)
    # Due to contention in the cluster, we may not be granted our full GPU demand
    # In this case, our demand should be a multiple of the number of GPUs assigned,
    # and this multiple is the number of virtual nodes per device
    if self.gpu_demand % num_assigned_gpus != 0:
      raise ValueError("GPU demand (%s) was not a multiple of " % self.gpu_demand +
        "number of GPUs assigned (%s) for job %s" % (num_assigned_gpus, job_id))
    env["NUM_VIRTUAL_NODES_PER_DEVICE"] = str(int(self.gpu_demand / num_assigned_gpus))
    env["NUM_GPUS"] = "1"
    env["BATCH_SIZE"] = str(self.batch_size)
    if self.num_steps is not None:
      env["NUM_STEPS"] = str(self.num_steps)
    if self.num_epochs is not None:
      env["NUM_EPOCHS"] = str(self.num_epochs)
    env["ENABLE_ELASTICITY"] = "true"
    env["RUN_TAG"] = "%s-scheduler_job%s" %\
      (scheduler.scheduler_mode.name, job_id)
    env.update(os.environ)
    return subprocess.Popen([EXECUTABLE, RUN_SCRIPT],\
      env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  def dummy_run(self, num_gpus):
    """
    Run with a fake scheduler, for testing purposes.
    """
    from collections import namedtuple
    from deploy.scheduler import SchedulerMode
    host = socket.gethostname()
    fake_scheduler = namedtuple("FakeScheduler",\
      ["addr", "scheduler_mode", "num_gpus_per_node", "gpu_assignment"])
    fake_scheduler = fake_scheduler(None, SchedulerMode.FIFO, num_gpus, {host: None})
    self.run(fake_scheduler, 0, host, num_gpus)

  def __str__(self):
    return "%s(gpu_demand=%s, batch_size=%s, num_steps=%s, num_epochs=%s)" %\
      (self.__class__.__name__, self.gpu_demand, self.batch_size, self.num_steps, self.num_epochs)

  def to_json(self):
    j = {"gpu_demand": self.gpu_demand, "batch_size": self.batch_size}
    if self.num_steps is not None: j["num_steps"] = self.num_steps
    if self.num_epochs is not None: j["num_epochs"] = self.num_epochs
    return j

  @classmethod
  def from_json(cls, j):
    return cls(j["gpu_demand"], j["batch_size"], j.get("num_steps"), j.get("num_epochs"))

  @staticmethod
  def name_to_cls():
    return {
      "resnet": ResNetWorkload,
      "bert": BERTGlueWorkload,
      "transformer": TransformerWorkload
    }

  @staticmethod
  def cls_to_name():
    return {v: k for k, v in Workload.name_to_cls().items()}

class ResNetWorkload(Workload):

  def __init__(self, dataset, gpu_demand, batch_size, num_steps=None, num_epochs=None):
    super(ResNetWorkload, self).__init__(gpu_demand, batch_size, num_steps, num_epochs)
    self.dataset = dataset

  def run(self, scheduler, job_id, master_host, num_assigned_gpus, env={}):
    env = env.copy()
    env["MODEL"] = "resnet"
    env["DATASET"] = self.dataset
    return super().run(scheduler, job_id, master_host, num_assigned_gpus, env)

  def __str__(self):
    return "ResNetWorkload(dataset=%s, gpu_demand=%s, batch_size=%s, num_steps=%s, num_epochs=%s)" %\
      (self.dataset, self.gpu_demand, self.batch_size, self.num_steps, self.num_epochs)

  def to_json(self):
    j = super().to_json()
    j["dataset"] = self.dataset
    return j

  @classmethod
  def from_json(cls, j):
    return ResNetWorkload(j["dataset"], j["gpu_demand"], j["batch_size"],
      j.get("num_steps"), j.get("num_epochs"))

class BERTGlueWorkload(Workload):

  def __init__(self, glue_task, gpu_demand, batch_size, num_steps=None, num_epochs=None):
    super(BERTGlueWorkload, self).__init__(gpu_demand, batch_size, num_steps, num_epochs)
    self.glue_task = glue_task

  def run(self, scheduler, job_id, master_host, num_assigned_gpus, env={}):
    env = env.copy()
    env["MODEL"] = "bert"
    env["BERT_TASK"] = "glue"
    env["GLUE_TASK"] = self.glue_task
    return super().run(scheduler, job_id, master_host, num_assigned_gpus, env)

  def __str__(self):
    return "BERTGlueWorkload(task=%s, gpu_demand=%s, batch_size=%s, num_steps=%s, num_epochs=%s)" %\
      (self.glue_task, self.gpu_demand, self.batch_size, self.num_steps, self.num_epochs)

  def to_json(self):
    j = super().to_json()
    j["glue_task"] = self.glue_task
    return j

  @classmethod
  def from_json(cls, j):
    return BERTGlueWorkload(j["glue_task"], j["gpu_demand"], j["batch_size"],
      j.get("num_steps"), j.get("num_epochs"))

class TransformerWorkload(Workload):

  def run(self, scheduler, job_id, master_host, num_assigned_gpus, env={}):
    env = env.copy()
    env["MODEL"] = "transformer"
    env["SKIP_EVAL"] = "true" # Decoding takes a long time
    return super().run(scheduler, job_id, master_host, num_assigned_gpus, env)

# Helper functions

def generate_schedule_from_trace(path):
  """
  Construct a list of 2-tuples (arrival_time, job) from a JSON trace.
  `arrival_time` refers to seconds elapsed after the scheduler started.
  """
  schedule = []
  with open(path) as f:
    for i, j in enumerate(json.load(f)):
      arrival_time = j["arrival_time"]
      workload_cls = Workload.name_to_cls()[j["workload"]]
      workload = workload_cls.from_json(j)
      priority = j.get("priority") or 1
      job = Job(i, workload, priority)
      schedule.append((arrival_time, job))
  return schedule

def poisson_interarrival_time(jobs_per_hour):
  """
  Return a random interarrival time based on the poisson distribution.
  """
  return int(-math.log(1.0 - random.random()) / (jobs_per_hour / 3600))

def generate_trace(trace_path, num_jobs, jobs_per_hour, workloads):
  """
  Generate a trace of jobs choosing from the given list of workloads.
  """
  with open(trace_path, "w") as f:
    f.write("[\n")
    cumulative_time = 0
    for i in range(num_jobs):
      workload = random.choice(workloads)
      cumulative_time += poisson_interarrival_time(jobs_per_hour)
      workload_name = Workload.cls_to_name()[workload.__class__]
      j = {}
      j["arrival_time"] = cumulative_time
      j["workload"] = workload_name
      j.update(workload.to_json())
      maybe_comma = "" if i == num_jobs-1 else ","
      f.write("  %s%s\n" % (json.dumps(j), maybe_comma))
    f.write("]\n")

def generate_tiny_workload_trace(trace_path, num_jobs, jobs_per_hour):
  """
  Generate a trace of tiny jobs, intended for testing.
  """
  resnet_dataset = "cifar10"
  glue_task = "MRPC"
  workloads = [
    ResNetWorkload(resnet_dataset, 1, 32, num_steps=500),
    ResNetWorkload(resnet_dataset, 2, 64, num_steps=500),
    ResNetWorkload(resnet_dataset, 4, 128, num_steps=500),
    ResNetWorkload(resnet_dataset, 1, 32, num_epochs=1),
    ResNetWorkload(resnet_dataset, 2, 64, num_epochs=1),
    ResNetWorkload(resnet_dataset, 4, 128, num_epochs=1),
    TransformerWorkload(1, 128, num_steps=50),
    TransformerWorkload(2, 256, num_steps=50),
    TransformerWorkload(4, 512, num_steps=50),
    TransformerWorkload(1, 128, num_steps=300),
    TransformerWorkload(2, 256, num_steps=300),
    TransformerWorkload(4, 512, num_steps=300),
    BERTGlueWorkload(glue_task, 1, 2, num_steps=50),
    BERTGlueWorkload(glue_task, 2, 4, num_steps=50),
    BERTGlueWorkload(glue_task, 4, 8, num_steps=50),
    BERTGlueWorkload(glue_task, 1, 2, num_steps=200),
    BERTGlueWorkload(glue_task, 2, 4, num_steps=200),
    BERTGlueWorkload(glue_task, 4, 8, num_steps=200)
  ]
  generate_trace(trace_path, num_jobs, jobs_per_hour, workloads)

def main():
  args = sys.argv
  if len(args) != 5:
    print("Usage: python3 scheduler_workload.py "
      "[trace_path] [num_jobs] [jobs_per_hour] [tiny|medium|big]")
    sys.exit(1)
  trace_path = args[1]
  num_jobs = int(args[2])
  jobs_per_hour = int(args[3])
  workload_size = args[4]
  if workload_size == "tiny":
    generate_tiny_workload_trace(trace_path, num_jobs, jobs_per_hour)
  elif workload_size == "medium" or workload_size == "big":
    raise ValueError("Workload size not supported yet: %s" % workload_size)
  else:
    raise ValueError("Unknown workload size: %s" % workload_size)
  print("Wrote to %s." % trace_path)

if __name__ == "__main__":
  main()

