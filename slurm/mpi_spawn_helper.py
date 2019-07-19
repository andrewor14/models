#!/usr/bin/env python3

import os

from mpi4py import MPI


# Environment variables
PYTHONPATH = "PYTHONPATH"
OMPI_MCA_orte_output_filename = "OMPI_MCA_orte_output_filename"
MPI_SPAWN_LOG_DIR = "MPI_SPAWN_LOG_DIR"
MPI_SPAWN_RANK = "MPI_SPAWN_RANK"
LAUNCH_SCRIPT_NAME = "LAUNCH_SCRIPT_NAME"

# Other constants
REDIRECT_SCRIPT_NAME = "redirect_mpi.sh"
THIS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXECUTABLE = "bash"

def spawn(spawn_comm, spawned_rank, launch_script=None, args=[], env={}):
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
  env[PYTHONPATH] = os.getenv(PYTHONPATH)
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
  # Set arguments, assuming the scripts are in the same directory as this file
  launch_script = launch_script or os.environ[LAUNCH_SCRIPT_NAME]
  launch_script = os.path.join(THIS_DIRECTORY, launch_script)
  redirect_script = os.path.join(THIS_DIRECTORY, REDIRECT_SCRIPT_NAME)
  args = [redirect_script, launch_script] + args
  return spawn_comm.Spawn(EXECUTABLE, args=args, info=info, maxprocs=1)

