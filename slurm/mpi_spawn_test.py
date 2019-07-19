import os
import sys
from mpi4py import MPI

from slurm import mpi_spawn_helper

JOINING_ENV_VAR = "JOINING"
SERVICE_NAME_ITERATION = "spawn_test_iteration"
STARTING_TAG = 100
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))

def log(msg):
  identifier = os.getenv(JOINING_ENV_VAR, "Master")
  identifier = "%s on %s" % (identifier, MPI.Get_processor_name())
  print("%s: %s" % (identifier, msg))

def main():
  try:
    algorithm(MPI.COMM_WORLD)
  except Exception as e:
    log("##### ERROR #####")
    raise e

def algorithm(comm):
  """
  Start the algorithm with a communicator that only includes the master
  then slowly spawn workers one by one and let them join our communicator.
  """
  is_joining = JOINING_ENV_VAR in os.environ
  is_master = comm.rank == 0 and not is_joining
  iteration = 1
  while comm.size < MAX_WORKERS:
    # Master will publish the iteration number
    if is_master:
      info = MPI.Info.Create()
      info.Set("ompi_global_scope", "true")
      MPI.Publish_name(SERVICE_NAME_ITERATION, str(iteration), info=info)

    # Wait for everyone to get the latest iteration number
    comm.barrier()
    iteration = int(MPI.Lookup_name(SERVICE_NAME_ITERATION))

    log("********** Join remote iteration %s **********" % iteration)

    if is_master or is_joining:
      if is_master:
        log("Master is spawning worker %s" % iteration)
        env = { JOINING_ENV_VAR: "Joined_%s" % iteration }
        spawn_intercomm = mpi_spawn_helper.spawn(MPI.COMM_SELF, iteration, env=env)
      else:
        spawn_intercomm = MPI.Comm.Get_parent()
      merged_intracomm = spawn_intercomm.Merge(is_joining)
    else:
      merged_intracomm = MPI.Intracomm(MPI.COMM_NULL)

    # Wait for everyone to get the tag
    comm.barrier()

    # Create an intercomm from merging comm with merge_intracomm
    tag = STARTING_TAG + iteration
    log("Waiting to super merge using tag %s" % tag)
    if is_joining:
      super_merged_intercomm = MPI.Intracomm.Create_intercomm(comm, 0, merged_intracomm, 0, tag)
    else:
      super_merged_intercomm = MPI.Intracomm.Create_intercomm(comm, 0, merged_intracomm, 1, tag)
    log("Super merged into intercomm")
    # Finally, convert this intercomm into an intracomm
    comm = super_merged_intercomm.Merge(is_joining)
    log("Our rank in new communicator = %s (size %s)" % (comm.rank, comm.size))

    # Clean up
    if is_master:
      MPI.Unpublish_name(SERVICE_NAME_ITERATION, str(iteration))

    # Try broadcasting a value
    value = "broadcast(%s)" % iteration if is_master else None
    value = comm.bcast(value, root=0)
    if not is_master:
      log("Received broadcast from master: %s" % value)

    is_joining = False
    iteration += 1

if __name__ == "__main__":
  main()

