import os
from mpi4py import MPI

JOINING_ENV_VAR = "JOINING"
SERVICE_NAME_PORT = "accept_test_port"
SERVICE_NAME_TAG = "accept_test_tag"
STARTING_TAG = 100
MAX_WORKERS = 10
RANK = MPI.COMM_WORLD.rank

def log(msg):
  print("%s: %s" % (RANK, msg))

def main():
  try:
    comm = MPI.COMM_WORLD
    if MPI.COMM_WORLD.size > 1:
      comm = join_local()
    algorithm(comm)
  except Exception as e:
    log("##### ERROR #####")
    raise e

def join_local():
  """
  Join everyone locally into an intracomm, should be the same as MPI.COMM_WORLD.
  """
  comm = MPI.COMM_SELF
  for i in range(1, MPI.COMM_WORLD.size):
    log("========== Join local iteration %s ==========" % i)
    if RANK <= i:
      if RANK < i:
        comm = MPI.Intracomm.Create_intercomm(comm, 0, MPI.COMM_WORLD, i, 100 + i)
      elif RANK == i:
        comm = MPI.Intracomm.Create_intercomm(MPI.COMM_SELF, 0, MPI.COMM_WORLD, 0, 100 + i)
      comm = comm.Merge(i > 0)
      # Test broadcast
      value = "broadcast(%s)" % i if RANK == 0 else None
      value = comm.bcast(value, root=0)
      if RANK != 0:
        log("Received broadcast value = %s" % value)
    # Wait for others
    MPI.COMM_WORLD.barrier() 
  return comm

def algorithm(comm):
  """
  Start the algorithm with a communicator that only includes the master
  then slowly add workers one by one through the Connect/Accept interface.
  """
  is_joining = JOINING_ENV_VAR in os.environ
  iteration = 1
  while iteration < MAX_WORKERS:
    log("********** Join remote iteration %s **********" % iteration)
    is_master = comm.rank == 0 and not is_joining

    # Master will open a port and publish it
    if is_master:
      port = MPI.Open_port()
      tag = STARTING_TAG + iteration
      info = MPI.Info.Create()
      info.Set("ompi_global_scope", "true")
      MPI.Publish_name(SERVICE_NAME_PORT, port, info=info)
      MPI.Publish_name(SERVICE_NAME_TAG, str(tag), info=info)

    # Connect master and the joining worker through that port
    if is_master or is_joining:
      if is_master:
        connect_intercomm = MPI.COMM_SELF.Accept(port)
      else:
        port = MPI.Lookup_name(SERVICE_NAME_PORT)
        connect_intercomm = MPI.COMM_SELF.Connect(port)
      merged_intracomm = connect_intercomm.Merge(is_joining)
      log("Our rank in merged_intracomm = %s (size %s)" % (merged_intracomm.rank, merged_intracomm.size))
    else:
      merged_intracomm = MPI.Intracomm(MPI.COMM_NULL)

    # Wait for everyone to get the tag
    comm.barrier()

    # Create an intercomm from merging comm with merge_intracomm
    tag = int(MPI.Lookup_name(SERVICE_NAME_TAG))
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
      MPI.Close_port(port)
      MPI.Unpublish_name(SERVICE_NAME_PORT, port)
      MPI.Unpublish_name(SERVICE_NAME_TAG, str(tag))
    if is_master or is_joining:
      connect_intercomm.Disconnect()

    # Try broadcasting a value
    value = "broadcast(%s)" % iteration if is_master else None
    value = comm.bcast(value, root=0)
    if not is_master:
      log("Received broadcast from master: %s" % value)

    is_joining = False
    iteration += 1

if __name__ == "__main__":
  main()

