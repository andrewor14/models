import os
import sys
import textwrap

from mpi4py import MPI
import horovod.tensorflow.keras as hvd

from official.resnet import mpi_helper


AUTOSCALING_MASTER_HOST_PORT = "AUTOSCALING_MASTER_HOST_PORT"
STARTING_TAG = 100
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))

def log(msg):
  identifier = os.getenv(AUTOSCALING_MASTER_HOST_PORT,\
    "Master" if MPI.COMM_WORLD.rank == 0 else "Worker")
  identifier = "%s on %s" % (identifier, MPI.Get_processor_name())
  print("%s: %s" % (identifier, msg))

def main():
  try:
    algorithm(MPI.COMM_WORLD.Dup())
  except Exception as e:
    log("##### ERROR #####")
    raise e

def algorithm(comm):
  """
  Start the algorithm with a communicator that only includes the root
  then slowly spawn workers one by one and let them join our communicator.
  """
  is_joining = AUTOSCALING_MASTER_HOST_PORT in os.environ
  is_root = comm.rank == 0 and not is_joining
  while comm.size < MAX_WORKERS:
    log("========== Join remote, current size = %s ==========" % comm.size)
    spawn_intercomm = None
    if is_root:
      log("Master is spawning worker %s" % comm.size)
      env = { AUTOSCALING_MASTER_HOST_PORT: "Joined worker(%s)" % comm.size }
      spawn_intercomm = mpi_helper.spawn(comm.size, env=env)
    elif is_joining:
      spawn_intercomm = MPI.Comm.Get_parent()
    comm = mpi_helper.expand(comm, spawn_intercomm)
    is_joining = False
  log(textwrap.dedent("""
    ***********************************************************
      All done, our rank in final communicator = %s (size %s)
    ***********************************************************""" % (comm.rank, comm.size)))

if __name__ == "__main__":
  main()

