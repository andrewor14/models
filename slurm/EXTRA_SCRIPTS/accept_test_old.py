from mpi4py import MPI

SERVICE_NAME_PORT = "accept_test_port"
SERVICE_NAME_TAG = "accept_test_tag"
STARTING_TAG = 100
WORLD_COMM = MPI.COMM_WORLD.Dup()
SIZE = WORLD_COMM.size
RANK = WORLD_COMM.rank

def log(msg):
  print("%s: %s" % (RANK, msg))

def main():
  try:
    algorithm()
  except Exception as e:
    log("##### ERROR #####")
    raise e

def algorithm():
  """
  Start the algorithm with a communicator that only includes the master
  then slowly add workers one by one through the Connect/Accept interface.
  """
  comm = MPI.COMM_SELF
  join_rank = 1
  while join_rank < SIZE:
    log("===== Iteration %s =====" % join_rank)
    is_master = RANK == 0
    is_joining = RANK == join_rank

    # Master will open a port and publish it
    if is_master:
      port = MPI.Open_port()
      tag = STARTING_TAG + join_rank
      info = MPI.Info.Create()
      info.Set("ompi_global_scope", "true")
      MPI.Publish_name(SERVICE_NAME_PORT, port, info=info)
      MPI.Publish_name(SERVICE_NAME_TAG, str(tag), info=info)

    # Add a synchronization barrier here to ensure the port is published before being accessed
    WORLD_COMM.barrier()
  
    # Connect master and the joining worker through that port
    if is_master or is_joining:
      if is_master:
        connect_intercomm = MPI.COMM_SELF.Accept(port)
      else:
        port = MPI.Lookup_name(SERVICE_NAME_PORT)
        connect_intercomm = MPI.COMM_SELF.Connect(port)
      merged_intracomm = connect_intercomm.Merge(is_joining)
      log("Connected master and %s" % join_rank)
    else:
      merged_intracomm = MPI.Intracomm(MPI.COMM_NULL)

    # Wait for everyone
    WORLD_COMM.barrier()
    tag = int(MPI.Lookup_name(SERVICE_NAME_TAG))
  
    # Create an intercomm from merging comm with merge_intracomm
    if RANK <= join_rank:
      if is_joining:
        super_merged_intercomm = MPI.Intracomm.Create_intercomm(comm, 0, merged_intracomm, 0, tag)
      else:
        super_merged_intercomm = MPI.Intracomm.Create_intercomm(comm, 0, merged_intracomm, 1, tag)
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

    join_rank += 1

if __name__ == "__main__":
  main()

