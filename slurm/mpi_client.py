# client.py

"""
Client side of the MPI client/server programming model.

Run this with 1 processes like:
$ mpiexec -n 1 python client.py
"""

import os
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD

# connect to the server
port_name = os.environ["MPI_OPEN_PORT_NAME"]
inter_comm = comm.Connect(port_name)

print("All good!")

# send message to the server
#send_obj = '1 + 2'
#print('Client sends %s to server.' % send_obj)
#inter_comm.send(send_obj, dest=0, tag=0)
## get results from the server
#recv_obj = inter_comm.recv(source=0, tag=1)
#print('Client receives %s from server.' % recv_obj)

# disconnect from the server
inter_comm.Disconnect()

print("Done")

