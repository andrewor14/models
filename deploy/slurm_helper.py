# --------------------------------------------------------------------
# Code derived from
# https://github.com/deepsense-ai/tensorflow_on_slurm
# --------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import re
import sys


SLURM_JOB_NODELIST = "SLURM_JOB_NODELIST"
SLURM_JOB_NUM_NODES = "SLURM_JOB_NUM_NODES"
SLURMD_NODENAME = "SLURMD_NODENAME"

# For multiplexing a node to multiple processes
SLURMD_PROC_INDEX = "SLURMD_PROC_INDEX"
SLURM_JOB_NUM_PROCS_PER_NODE = "SLURM_JOB_NUM_PROCS_PER_NODE"

def running_through_slurm():
  return SLURM_JOB_NODELIST in os.environ and SLURMD_NODENAME in os.environ

def set_tf_config(num_ps, port_number=2222):
  """
  Set TF_CONFIG to configurations read from slurm environment variables.
  """
  cluster, my_job_name, my_task_index, _ = tf_config_from_slurm(num_ps, port_number)
  tf_config_dict = {"cluster": cluster, "task": {"type": my_job_name, "index": my_task_index}}
  os.environ["TF_CONFIG"] = json.dumps(tf_config_dict)

def tf_config_from_slurm(num_ps, port_number=2222):
  """
  Creates configuration for a distributed tensorflow session
  from environment variables provided by the Slurm cluster
  management system.

  Note: This assumes that nodes are either ps or workers,
  and so does not work with the estimator API.

  @param: num_ps number of parameter servers to run
  @param: port_number port number to be used for communication
  @return: a 4-tuple (cluster_spec, task_name, task_index, host_port)
  """

  if not running_through_slurm():
    raise ValueError("Slurm environment variables not found.")

  node_name = os.environ[SLURMD_NODENAME]
  node_list = _expand_node_list(os.environ[SLURM_JOB_NODELIST])
  num_nodes = int(os.environ[SLURM_JOB_NUM_NODES])
  proc_index = int(os.getenv(SLURMD_PROC_INDEX) or 0)
  num_procs_per_node = int(os.getenv(SLURM_JOB_NUM_PROCS_PER_NODE) or 1)

  if num_procs_per_node > 4:
    raise ValueError("We currently don't support more than 4 processes on one node")

  if len(node_list) != num_nodes:
    raise ValueError("Number of slurm nodes {} not equal to {}"
                     .format(len(node_list), num_nodes))

  if node_name not in node_list:
    raise ValueError("Node name ({}) not in node list ({}). This should not happen! "
                     .format(node_name, node_list))

  if proc_index < 0 or proc_index >= num_procs_per_node:
    raise ValueError("{} must be between 0 and {}, was {}"
                     .format(SLURMD_PROC_INDEX, num_procs_per_node - 1, proc_index))

  # Attach the port number to each node and maybe expand each node into multiple processes
  new_node_list = []
  for node in node_list:
    for i in range(num_procs_per_node):
      new_node_name = "%s:%s" % (node, port_number + i)
      new_node_list.append(new_node_name)
      if node == node_name and i == proc_index:
        node_name = new_node_name
  node_list = new_node_list

  # Assign parameter servers and workers
  ps_nodes = [node for i, node in enumerate(node_list) if i < num_ps]
  worker_nodes = [node for i, node in enumerate(node_list) if i >= num_ps]

  if node_name in ps_nodes:
    my_job_name = "ps"
    my_task_index = ps_nodes.index(node_name)
  elif node_name in worker_nodes:
    my_job_name = "worker"
    my_task_index = worker_nodes.index(node_name)
  else:
    raise ValueError("Node name ({}) is neither a ps nor a worker!".format(node_name))

  if num_ps > 0:
    cluster = {"ps": ps_nodes, "worker": worker_nodes}
  else:
    cluster = {"worker": worker_nodes}

  return cluster, my_job_name, my_task_index, node_name

def _expand_node_list(node_list):
  """
  Expand a list of comma-separated nodes in shortened slurm format.

  In the node list to be expanded, nodes sharing the same prefix can be grouped together,
  with the different suffixes listed in square brackets following the prefix. For example,
  a node list of "tiger-i19g7,tiger-i20g[1,5-7]" expands to five nodes:

    tiger-i19g7
    tiger-i20g1
    tiger-i20g5
    tiger-i20g6
    tiger-i20g7

  The return value for this function should be the same as running:

    scontrol show hostname $SLURM_JOB_NODELIST

  Since `scontrol` may not be available on all systems, we will re-implement the expansion
  here in python ourselves.

  :param node_list: list of slurm node in shortened format
  :return: list of strings, each one is a hostname
  """
  #import subprocess
  #return subprocess.run(
  #    ["scontrol show hostname $SLURM_JOB_NODELIST"],
  #    shell=True,
  #    stdout=subprocess.PIPE).stdout.decode('utf-8').split()

  nodes = []
  patterns = []

  # First, split the node list into patterns, e.g. ["tiger-i19g7", "tiger-i20g[1,5-7]"].
  # Note that we cannot just use split because some commas exist within square brackets.
  pattern_so_far = ""
  in_square_brackets = False
  for c in node_list:
    if c == "[":
      in_square_brackets = True
    if c == "]":
      in_square_brackets = False
    if c == "," and not in_square_brackets:
      patterns.append(pattern_so_far)
      pattern_so_far = ""
    else:
      pattern_so_far += c
  patterns.append(pattern_so_far)

  # Next, we break down each pattern
  for pattern in patterns:
    # Look for square brackets
    m = re.match("(.*)\[(.*)\]", pattern)
    if m is None:
      # This is just one node, so we collect it
      nodes.append(pattern)
    else:
      # This is actually multiple nodes
      # e.g. prefix = "tiger-i20g", suffix = "1,5-7"
      (prefix, suffixes) = m.groups()
      for group in suffixes.split(","):
        m = re.match("(\d*)\-(\d*)", group)
        # e.g. group = "5-7"
        if m is not None:
          (start, end) = m.groups()
          for i in range(int(start), int(end) + 1):
            nodes.append("%s%s" % (prefix, i))
        # e.g. group = "1"
        elif group.isdigit():
          nodes.append("%s%s" % (prefix, group))
        else:
          raise ValueError("Unexpected pattern '%s' in node list '%s'" % (group, node_list))
  return nodes

if __name__ == "__main__":
  import sys
  args = sys.argv
  if len(args) != 2:
    print("Usage: tensorflow_on_slurm.py [node list]")
    sys.exit(1)
  for node in _expand_node_list(args[1]):
    print(node)

