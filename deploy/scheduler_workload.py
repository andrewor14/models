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

  def __init__(self,
      gpu_demand,
      batch_size,
      num_steps=None,
      num_epochs=None,
      num_virtual_nodes_per_device=None):
    self.gpu_demand = gpu_demand
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.num_epochs = num_epochs
    self.num_virtual_nodes_per_device = num_virtual_nodes_per_device or 1

  def run(self, scheduler, job_id, master_host, num_assigned_gpus, env={}):
    """
    Run this workload asynchronously and return the process object.
    This assumes the caller holds `scheduler.lock`.
    """
    job_env = {}
    job_env["JOB_ID"] = str(job_id)
    if scheduler.addr is not None:
      job_env["SCHEDULER_ADDR"] = scheduler.addr
    job_env["MASTER_HOST"] = master_host
    job_env["MPI_HOSTS"] = ",".join(list(scheduler.gpu_assignment.keys()).copy())
    job_env["NUM_GPUS_PER_NODE"] = str(scheduler.num_gpus_per_node)
    job_env["NUM_NODES"] = str(num_assigned_gpus)
    # Due to contention in the cluster, we may not be granted our full GPU demand
    # In this case, our demand should be a multiple of the number of GPUs assigned
    # so that the number of virtual nodes will be an integer
    if self.gpu_demand % num_assigned_gpus != 0:
      raise ValueError("GPU demand (%s) was not a multiple of " % self.gpu_demand +
        "number of GPUs assigned (%s) for job %s" % (num_assigned_gpus, job_id))
    job_env["NUM_VIRTUAL_NODES_PER_DEVICE"] =\
      str(int(self.num_virtual_nodes_per_device * self.gpu_demand / num_assigned_gpus))
    job_env["NUM_GPUS"] = "1"
    job_env["BATCH_SIZE"] = str(self.batch_size)
    if self.num_steps is not None:
      job_env["NUM_STEPS"] = str(self.num_steps)
    if self.num_epochs is not None:
      job_env["NUM_EPOCHS"] = str(self.num_epochs)
    job_env["ENABLE_ELASTICITY"] = "true"
    job_env["ENABLE_XLA"] = "false"
    job_env["RUN_TAG"] = "%s-scheduler_job%s" %\
      (scheduler.scheduler_mode.name, job_id)
    env = env.copy()
    env.update(os.environ)
    env.update(job_env)
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
    if self.num_steps is not None:
      j["num_steps"] = self.num_steps
    if self.num_epochs is not None:
      j["num_epochs"] = self.num_epochs
    if self.num_virtual_nodes_per_device > 1:
      j["num_virtual_nodes_per_device"] = self.num_virtual_nodes_per_device
    return j

  @classmethod
  def from_json(cls, j):
    return cls(j["gpu_demand"], j["batch_size"], j.get("num_steps"),
      j.get("num_epochs"), j.get("num_virtual_nodes_per_device"))

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

  def __init__(self,
      dataset,
      gpu_demand,
      batch_size,
      num_steps=None,
      num_epochs=None,
      num_virtual_nodes_per_device=None):
    super(ResNetWorkload, self).__init__(gpu_demand, batch_size,
      num_steps, num_epochs, num_virtual_nodes_per_device)
    self.dataset = dataset

  def run(self, scheduler, job_id, master_host, num_assigned_gpus, env={}):
    env = env.copy()
    env["MODEL"] = "resnet"
    env["DATASET"] = self.dataset
    return super().run(scheduler, job_id, master_host, num_assigned_gpus, env)

  def __str__(self):
    return ("ResNetWorkload(dataset=%s, gpu_demand=%s, batch_size=%s, "\
      "num_steps=%s, num_epochs=%s, num_virtual_nodes_per_device=%s)") %\
       (self.dataset, self.gpu_demand, self.batch_size, self.num_steps,\
       self.num_epochs, self.num_virtual_nodes_per_device)

  def to_json(self):
    j = super().to_json()
    j["dataset"] = self.dataset
    return j

  @classmethod
  def from_json(cls, j):
    return ResNetWorkload(j["dataset"], j["gpu_demand"], j["batch_size"],
      j.get("num_steps"), j.get("num_epochs"), j.get("num_virtual_nodes_per_device"))

class BERTGlueWorkload(Workload):

  def __init__(self,
      glue_task,
      gpu_demand,
      batch_size,
      num_steps=None,
      num_epochs=None,
      num_virtual_nodes_per_device=None):
    super(BERTGlueWorkload, self).__init__(gpu_demand, batch_size,
      num_steps, num_epochs, num_virtual_nodes_per_device)
    self.glue_task = glue_task

  def run(self, scheduler, job_id, master_host, num_assigned_gpus, env={}):
    env = env.copy()
    env["MODEL"] = "bert"
    env["BERT_TASK"] = "glue"
    env["GLUE_TASK"] = self.glue_task
    return super().run(scheduler, job_id, master_host, num_assigned_gpus, env)

  def __str__(self):
    return ("BERTGlueWorkload(task=%s, gpu_demand=%s, batch_size=%s, "\
      "num_steps=%s, num_epochs=%s, num_virtual_nodes_per_device=%s)") %\
       (self.glue_task, self.gpu_demand, self.batch_size, self.num_steps,\
       self.num_epochs, self.num_virtual_nodes_per_device)

  def to_json(self):
    j = super().to_json()
    j["glue_task"] = self.glue_task
    return j

  @classmethod
  def from_json(cls, j):
    return BERTGlueWorkload(j["glue_task"], j["gpu_demand"], j["batch_size"],
      j.get("num_steps"), j.get("num_epochs"), j.get("num_virtual_nodes_per_device"))

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

def generate_trace(trace_path, num_jobs, jobs_per_hour, priorities, workload_size):
  """
  Generate a trace of jobs choosing from the given list of workloads.
  """
  workloads = []
  if workload_size == "tiny":
    workloads = tiny_workload()
  elif workload_size == "medium":
    workloads = medium_workload()
  elif workload_size == "big":
    raise ValueError("Workload size not supported yet: %s" % workload_size)
  else:
    raise ValueError("Unknown workload size: %s" % workload_size)

  with open(trace_path, "w") as f:
    f.write("[\n")
    arrival_time_seconds = 1
    for i in range(num_jobs):
      workload = random.choice(workloads)
      workload_name = Workload.cls_to_name()[workload.__class__]
      j = {}
      j["arrival_time"] = arrival_time_seconds
      j["workload"] = workload_name
      j["priority"] = random.choice(priorities)
      j.update(workload.to_json())
      maybe_comma = "" if i == num_jobs-1 else ","
      f.write("  %s%s\n" % (json.dumps(j), maybe_comma))
      arrival_time_seconds += poisson_interarrival_time(jobs_per_hour)
    f.write("]\n")

def tiny_workload(resnet_dataset="cifar10", glue_task="MRPC"):
  """
  Return a list of tiny workloads, intended for testing.
  """
  return [
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

def medium_workload():
  """
  Return a list of medium workloads, intended for an 8x V100 GPU machine.
  """
  return [
    ResNetWorkload("cifar10", 1, 64, num_epochs=10), # 4 mins
    ResNetWorkload("cifar10", 1, 128, num_epochs=10), # 4 mins
    ResNetWorkload("cifar10", 1, 64, num_epochs=20), # 8 mins
    ResNetWorkload("cifar10", 1, 128, num_epochs=20), # 8 mins
    ResNetWorkload("imagenet", 1, 256, num_epochs=1), # 18 mins
    ResNetWorkload("imagenet", 2, 512, num_epochs=1), # 10 mins
    ResNetWorkload("imagenet", 4, 1024, num_epochs=2), # 8 mins
    ResNetWorkload("imagenet", 8, 2048, num_epochs=2), # 4 mins
    ResNetWorkload("imagenet", 1, 512, num_epochs=1, num_virtual_nodes_per_device=2), # 18 mins
    ResNetWorkload("imagenet", 2, 1024, num_epochs=1, num_virtual_nodes_per_device=2), # 10 mins
    ResNetWorkload("imagenet", 4, 2048, num_epochs=2, num_virtual_nodes_per_device=2), # 12 mins
    ResNetWorkload("imagenet", 8, 4096, num_epochs=3, num_virtual_nodes_per_device=2), # 9 mins
    ResNetWorkload("imagenet", 1, 1024, num_epochs=1, num_virtual_nodes_per_device=4), # 18 mins
    ResNetWorkload("imagenet", 2, 2048, num_epochs=1, num_virtual_nodes_per_device=4), # 10 mins
    ResNetWorkload("imagenet", 4, 4096, num_epochs=2, num_virtual_nodes_per_device=4), # 12 mins
    ResNetWorkload("imagenet", 8, 8192, num_epochs=3, num_virtual_nodes_per_device=4), # 9 mins
    BERTGlueWorkload("CoLA", 1, 8, num_epochs=10), # 15 mins
    BERTGlueWorkload("CoLA", 2, 16, num_epochs=10), # 11 mins
    BERTGlueWorkload("CoLA", 4, 32, num_epochs=10), # 10 mins
    BERTGlueWorkload("CoLA", 8, 64, num_epochs=10), # 10 mins
    BERTGlueWorkload("CoLA", 1, 16, num_epochs=10, num_virtual_nodes_per_device=2), # 15 mins
    BERTGlueWorkload("CoLA", 2, 32, num_epochs=10, num_virtual_nodes_per_device=2), # 11 mins
    BERTGlueWorkload("CoLA", 4, 64, num_epochs=20, num_virtual_nodes_per_device=2), # 20 mins
    BERTGlueWorkload("CoLA", 8, 128, num_epochs=20, num_virtual_nodes_per_device=2), # 19 mins
    BERTGlueWorkload("SST-2", 1, 8, num_epochs=1), # 10 mins
    BERTGlueWorkload("SST-2", 2, 16, num_epochs=1), # 9 mins
    BERTGlueWorkload("SST-2", 4, 32, num_epochs=1), # 8 mins
    BERTGlueWorkload("SST-2", 8, 64, num_epochs=1), # 7 mins
    BERTGlueWorkload("SST-2", 1, 16, num_epochs=1, num_virtual_nodes_per_device=2), # 10 mins
    BERTGlueWorkload("SST-2", 2, 32, num_epochs=1, num_virtual_nodes_per_device=2), # 9 mins
    BERTGlueWorkload("SST-2", 4, 64, num_epochs=2, num_virtual_nodes_per_device=2), # 16 mins
    BERTGlueWorkload("SST-2", 8, 128, num_epochs=2, num_virtual_nodes_per_device=2), # 14 mins
    TransformerWorkload(1, 4096, num_steps=500), # 4 mins
    TransformerWorkload(2, 8192, num_steps=500), # 4 mins
    TransformerWorkload(4, 16384, num_steps=500), # 6 mins
    TransformerWorkload(8, 32768, num_steps=500), # 5 mins
    TransformerWorkload(1, 4096, num_steps=1000), # 8 mins
    TransformerWorkload(2, 8192, num_steps=1000), # 8 mins
    TransformerWorkload(4, 16384, num_steps=1000), # 12 mins
    TransformerWorkload(8, 32768, num_steps=1000), # 9 mins
    TransformerWorkload(1, 8192, num_steps=500, num_virtual_nodes_per_device=2), # 8 mins
    TransformerWorkload(2, 16384, num_steps=500, num_virtual_nodes_per_device=2), # 8 mins
    TransformerWorkload(4, 32768, num_steps=500, num_virtual_nodes_per_device=2), # 12 mins
    TransformerWorkload(8, 65536, num_steps=500, num_virtual_nodes_per_device=2), # 9 mins
    TransformerWorkload(1, 8192, num_steps=1000, num_virtual_nodes_per_device=2), # 16 mins
    TransformerWorkload(2, 16384, num_steps=1000, num_virtual_nodes_per_device=2), # 16 mins
    TransformerWorkload(4, 32768, num_steps=1000, num_virtual_nodes_per_device=2), # 24 mins
    TransformerWorkload(8, 65536, num_steps=1000, num_virtual_nodes_per_device=2), # 18 mins
  ]

def main():
  args = sys.argv
  if len(args) != 6:
    print("Usage: python3 scheduler_workload.py "
      "[trace_path] [num_jobs] [jobs_per_hour] [priorities] [workload_size]")
    print("  priorities: comma-delimited list of priorities, e.g. 1,2,3")
    print("  workload_size: choose from 'tiny', 'medium', or 'big'")
    sys.exit(1)
  trace_path = args[1]
  num_jobs = int(args[2])
  jobs_per_hour = int(args[3])
  priorities = [int(p) for p in args[4].split(",")]
  workload_size = args[5]
  generate_trace(trace_path, num_jobs, jobs_per_hour, priorities, workload_size)
  print("Wrote to %s." % trace_path)

if __name__ == "__main__":
  main()

