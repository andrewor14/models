from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
membership_key = rank % 4

my_comm = MPI.COMM_WORLD.Split(membership_key, rank)
comm1 = None
comm2 = None

def log(msg):
  print("%s: %s" % (rank, msg))

comm = MPI.COMM_SELF
for i in range(1, MPI.COMM_WORLD.size):
  log("========== Iteration %s ==========" % i)
  if rank <= i:
    if rank < i:
      comm = MPI.Intracomm.Create_intercomm(comm, 0, MPI.COMM_WORLD, i, 100 + i)
    elif rank == i:
      comm = MPI.Intracomm.Create_intercomm(MPI.COMM_SELF, 0, MPI.COMM_WORLD, 0, 100 + i)
    comm = comm.Merge(i > 0)
    # Test broadcast
    value = "broadcast(%s)" % i if rank == 0 else None
    value = comm.bcast(value, root=0)
    if rank != 0:
      log("Received broadcast value = %s" % value)
  # Wait for others
  MPI.COMM_WORLD.barrier() 

## Build the communicators
#if membership_key == 0:
#  comm1 = MPI.Intracomm.Create_intercomm(my_comm, 0, MPI.COMM_WORLD, 1, 101)
#elif membership_key == 1:
#  comm1 = MPI.Intracomm.Create_intercomm(my_comm, 0, MPI.COMM_WORLD, 0, 101)
#elif membership_key == 2:
#  comm1 = MPI.Intracomm.Create_intercomm(my_comm, 0, MPI.COMM_WORLD, 3, 102)
#elif membership_key == 3:
#  comm1 = MPI.Intracomm.Create_intercomm(my_comm, 0, MPI.COMM_WORLD, 2, 102)
#
## Broadcast, let 0 and 2 become respective leaders of their groups
#comm1 = comm1.Merge(membership_key != 0 and membership_key != 2)
#value = None
#if rank == 0: value = "g1"
#if rank == 2: value = "g2"
#value = comm1.bcast(value, root=0)
#log("Value is %s" % value)
#
## Now let's create an intercomm out of the two intracomm groups
#log("Merging the two intracomms, should include everyone now")
#if membership_key < 2:
#  comm2 = MPI.Intracomm.Create_intercomm(comm1, 0, MPI.COMM_WORLD, 2, 103)
#else:
#  comm2 = MPI.Intracomm.Create_intercomm(comm1, 0, MPI.COMM_WORLD, 0, 103)
#
## Broadcast, let 0 become the leader
#comm2 = comm2.Merge(membership_key != 0)
#value = "g3" if rank == 0 else None
#value = comm2.bcast(value, root=0)
#log("Value is %s" % value)

