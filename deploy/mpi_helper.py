#!/usr/bin/env python3

import json
import os

from mpi4py import MPI
import tensorflow as tf
import horovod.tensorflow as hvd

from autoscaling.params import *


# Environment variables
PYTHONPATH = "PYTHONPATH"
MODELS_DIR = "MODELS_DIR"
OMPI_MCA_orte_output_filename = "OMPI_MCA_orte_output_filename"
OMPI_MCA_initial_wdir = "OMPI_MCA_initial_wdir"
MPI_SPAWN_LOG_DIR = "MPI_SPAWN_LOG_DIR"
MPI_SPAWN_RANK = "MPI_SPAWN_RANK"
MPI_TAG_NAME = "MPI_TAG_NAME"
LAUNCH_SCRIPT_NAME = "LAUNCH_SCRIPT_NAME"
TF_CONFIG = "TF_CONFIG"

# Other constants
LAUNCH_DIRECTORY = os.environ[OMPI_MCA_initial_wdir]
REDIRECT_SCRIPT_NAME = "redirect_mpi.sh"
EXECUTABLE = "bash"

# Tag for merging communicators, only used at the root
MPI_CURRENT_TAG = 14444

def log_fn(msg):
  tf.logging.info("[MPI helper]: %s" % msg)

def set_tf_config(base_port=2222):
  """
  Set TF_CONFIG based on hostnames of all processes in MPI.COMM_WORLD.
  To avoid port collisions, we add a process' rank to its port.
  """
  my_host = MPI.Get_processor_name()
  my_index = MPI.COMM_WORLD.rank
  if MPI.COMM_WORLD.size == 1:
    host_ports = ["%s:%s" % (my_host, base_port + int(os.getenv(MPI_SPAWN_RANK, 0)))]
  else:
    all_hosts = MPI.COMM_WORLD.allgather(my_host)
    host_ports = ["%s:%s" % (host, base_port + i) for i, host in enumerate(all_hosts)]
  tf_config = {"cluster": {"worker": host_ports}, "task": {"type": "worker", "index": my_index}}
  tf_config = json.dumps(tf_config)
  log_fn("Setting %s to %s" % (TF_CONFIG, tf_config))
  os.environ[TF_CONFIG] = tf_config

def expand(intracomm, intercomm=None):
  """
  Expand an existing intracommunicator by merging an intercommunicator into it.
  All members of both communicators must participate in this process.

  For example, the intercommunicator may be set to the following:
    - At the root: the communicator returned by MPI.Comm.Spawn
    - At the spawned worker: the communicator returned by MPI.Comm.Get_parent
    - At all other nodes: None

  Return a merged intracommunicator ready to be passed into `hvd.init`.
  """
  global MPI_CURRENT_TAG
  is_joining = intracomm.rank == 0 and AUTOSCALING_MASTER_HOST_PORT in os.environ
  is_root = intracomm.rank == 0 and not is_joining  
  tag = MPI_CURRENT_TAG if is_root else None

  if is_joining:
    log_fn("Joining an existing communicator")
  else:
    log_fn("Expanding communicator %s (current size %s)" % (intracomm, intracomm.size))

  # Merge the two intercommunicators into an intracommunicator
  if intercomm is not None:
    merged_intracomm = intercomm.Merge(is_joining)
  else:
    merged_intracomm = MPI.Intracomm(MPI.COMM_NULL)

  # The root broadcasts its tag in both communicators to make sure everyone has the same tag
  if intercomm is not None:
    tag = merged_intracomm.bcast(tag, root=0)
  tag = intracomm.bcast(tag, root=0)

  # Merge the two intracommunicators into an intercommunicator
  log_fn("Merging communicators using tag %s" % tag)
  if is_joining:
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(intracomm, 0, merged_intracomm, 0, tag)
  else:
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(intracomm, 0, merged_intracomm, 1, tag)

  # Finally, convert this intercommunicator into an intracommunicator
  comm = super_merged_intercomm.Merge(is_joining)
  log_fn("Our rank in new communicator = %s (size %s)" % (comm.rank, comm.size))

  # Run some collective operations on this communicator
  test_communication(comm)

  if is_root:
    MPI_CURRENT_TAG += 1
  return comm

def spawn(spawned_rank, launch_script=None, target_host=None, args=[], env={}):
  """
  Spawn a process using the given MPI communicator.

  This is a helper for spawning a process that will be merged into an existing communicator.
  The caller must provide `spawned_rank`, which specifies the expected rank of the spawned
  process in the new merged communicator.

  This method assumes the caller was launched using `mpirun` with either the `--host` or the
  `--hostfile` option, which controls the node on which the new process will be launched.
  """
  # Set environment variables
  env = env.copy()
  env[PYTHONPATH] = os.getenv(MODELS_DIR)
  env[MPI_SPAWN_LOG_DIR] = os.getenv(OMPI_MCA_orte_output_filename)
  env[MPI_SPAWN_RANK] = spawned_rank
  # Note: there is a max character limit for the value of MPI.Info!
  # Here we take care not to exceed it, otherwise we will see MPI_ERR_INFO_VALUE...
  env = "\n".join(["%s=%s" % (k, v) for k, v in env.items() if v is not None])
  if len(env) > MPI.MAX_INFO_VAL:
    raise ValueError("MPI environment string is longer than MPI_MAX_INFO_VAL(%s):\n%s" %\
      (MPI.MAX_INFO_VAL, env))
  info = MPI.Info.Create()
  info.Set("env", env)
  # Setting "bind_to" to "none" (default was "core") significantly improves MPI performance
  # for multi-threaded applications. See https://www.open-mpi.org/doc/v1.8/man1/mpirun.1.php
  info.Set("bind_to", "none")
  if target_host is not None:
    log_fn("Launching process on target host %s" % target_host)
    info.Set("host", target_host)
  # Set arguments, assuming the scripts are in the same directory as this file
  launch_script = launch_script or os.environ[LAUNCH_SCRIPT_NAME]
  launch_script = os.path.join(LAUNCH_DIRECTORY, launch_script)
  redirect_script = os.path.join(LAUNCH_DIRECTORY, REDIRECT_SCRIPT_NAME)
  args = [redirect_script, launch_script] + args
  return MPI.COMM_SELF.Spawn(EXECUTABLE, args=args, info=info, maxprocs=1)

def test_communication(comm):
  """
  Helper method to make sure basic communication works in the given communicator.
  """
  log_fn("Testing communication in communicator %s (size %s)" % (comm, comm.size))
  # Try broadcasting a value
  value = "[root value]" if comm.rank == 0 else None
  value = comm.bcast(value, root=0)
  if comm.rank > 0:
    log_fn("  Received broadcast from root: %s" % value)
  # Try doing an allreduce
  value = comm.allreduce(comm.rank, op=MPI.SUM)
  log_fn("  Allreduce result: %s" % value)

