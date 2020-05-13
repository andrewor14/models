#!/usr/bin/env python3

import os

from absl import logging
from mpi4py import MPI

from virtual.virtual_helper import *


EXECUTABLE = "bash"
LAUNCH_SCRIPT_NAME = "run_distributed.sh"
REDIRECT_SCRIPT_NAME = "redirect_mpi.sh"
MPI_CURRENT_TAG = 14444

def expand_communicator(intracomm, intercomm=None):
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
    logging.info("Joining an existing communicator")
  else:
    logging.info("Expanding communicator %s (current size %s)" % (intracomm, intracomm.size))

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
  logging.info("Merging communicators using tag %s" % tag)
  if is_joining:
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(intracomm, 0, merged_intracomm, 0, tag)
  else:
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(intracomm, 0, merged_intracomm, 1, tag)

  # Finally, convert this intercommunicator into an intracommunicator
  comm = super_merged_intercomm.Merge(is_joining)
  logging.info("Our rank in new communicator = %s (size %s)" % (comm.rank, comm.size))

  # Run some collective operations on this communicator
  test_communication(comm)

  if is_root:
    MPI_CURRENT_TAG += 1
  return comm

def spawn_process(spawn_rank, args=[], env={}):
  """
  Spawn a process and assign it an expected rank of `spawn_rank`.

  This is a helper for spawning a process that will be merged into an existing communicator.
  The caller must provide `spawn_rank`, which specifies the expected rank of the spawned
  process in the new merged communicator.

  This method assumes the caller was launched using `mpirun` with either the `--host` or the
  `--hostfile` option, which controls the node on which the new process will be launched.
  """
  # Set environment variables
  env = env.copy()
  env[PYTHONPATH] = os.getenv(MODELS_DIR)
  env[MPI_SPAWN_LOG_DIR] = os.getenv(OMPI_MCA_orte_output_filename)
  env[MPI_SPAWN_RANK] = spawn_rank
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
  # Set launch scripts
  launch_script = os.path.join(os.environ[DEPLOY_DIR], LAUNCH_SCRIPT_NAME)
  redirect_script = os.path.join(os.environ[DEPLOY_DIR], REDIRECT_SCRIPT_NAME)
  args = [redirect_script, launch_script] + args
  args = ["/home/andrewor/models/deploy/try/spawned.sh"]
  logging.info("Right before spawn, args = %s" % args)
  c = MPI.COMM_SELF.Spawn(EXECUTABLE, args=args, info=info, maxprocs=1)
  logging.info("Right after spawn")
  return c


def test_communication(comm):
  """
  Helper method to make sure basic communication works in the given communicator.
  """
  logging.info("Testing communication in communicator %s (size %s)" % (comm, comm.size))
  # Try broadcasting a value
  value = "[root value]" if comm.rank == 0 else None
  value = comm.bcast(value, root=0)
  if comm.rank > 0:
    logging.info("  Received broadcast from root: %s" % value)
  # Try doing an allreduce
  value = comm.allreduce(comm.rank, op=MPI.SUM)
  logging.info("  Allreduce result: %s" % value)

