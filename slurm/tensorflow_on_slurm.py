# --------------------------------------------------------------------
# Code derived from
# https://github.com/deepsense-ai/tensorflow_on_slurm
# --------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os


SLURM_JOB_NODELIST = "SLURM_JOB_NODELIST"
SLURM_JOB_NUM_NODES = "SLURM_JOB_NUM_NODES"
SLURMD_NODENAME = "SLURMD_NODENAME"

def running_through_slurm():
  return SLURM_JOB_NODELIST in os.environ and SLURMD_NODENAME in os.environ

def json_tf_config_from_slurm(ps_number, port_number=2222):
  """
  Creates json configuration, which will be assigned to environment variable TF_CONFIG
  to be used by Estimator for distributed training.

  Examples of roles and node indices can be found here:
  https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate

  Note that the evaluator node does not appear in cluster json string.

  :param ps_number: number of parameter server nodes
  :param port_number: port number to be used for communication
  :return: json string, containing cluster spec, as well as the role and index of a node.
  """

  cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number, port_number)
  tf_config_dict = {"cluster": cluster, "task": {"type": my_job_name, "index": my_task_index}}
  return json.dumps(tf_config_dict)


def tf_config_from_slurm(ps_number, port_number=2222):
  """
  Creates configuration for a distributed tensorflow session
  from environment variables  provided by the Slurm cluster
  management system.

  Updated by haoyuz:
  In order to work with TensorFlow Estimator API, nodes are divided into four different roles:
  1) Chief worker, only one
  2) Workers
  3) Parameter servers
  4) Evaluator, only one
  See https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate

  @param: ps_number number of parameter servers to run
  @param: port_number port number to be used for communication
  @return: a tuple containing cluster with fields cluster_spec,
           task_name and task_id
  """

  if not running_through_slurm():
    raise ValueError("Slurm environment variables not found.")

  nodename = os.environ[SLURMD_NODENAME]
  nodelist = _expand_nodelist(os.environ[SLURM_JOB_NODELIST])
  num_nodes = int(os.environ[SLURM_JOB_NUM_NODES])

  if len(nodelist) != num_nodes:
    raise ValueError("Number of slurm nodes {} not equal to {}".format(len(nodelist), num_nodes))

  if nodename not in nodelist:
    raise ValueError("Nodename({}) not in nodelist({}). This should not happen! ".format(nodename, nodelist))

  ps_nodes = [node for i, node in enumerate(nodelist) if i < ps_number]
  chief_node = [node for i, node in enumerate(nodelist) if i == ps_number]
  worker_nodes = [node for i, node in enumerate(nodelist) if i > ps_number and i < len(nodelist)-1]
  # Leave the last node unassigned, being the evaluator node.

  if nodename in ps_nodes:
    my_job_name = "ps"
    my_task_index = ps_nodes.index(nodename)
  elif nodename in chief_node:
    my_job_name = "chief"
    my_task_index = 0  # only one chief worker node
  elif nodename in worker_nodes:
    my_job_name = "worker"
    my_task_index = worker_nodes.index(nodename)
  else:
    my_job_name = "evaluator"
    my_task_index = 0  # only one evaluator node

  ps_ipports = [":".join([node, str(port_number)]) for node in ps_nodes]
  chief_ipports = [":".join([node, str(port_number)]) for node in chief_node]
  worker_ipports  = [":".join([node, str(port_number)]) for node in worker_nodes]
  cluster = {"chief": chief_ipports, "worker": worker_ipports, "ps": ps_ipports}

  return cluster, my_job_name, my_task_index


def _pad_zeros(iterable, length):
  return (str(t).rjust(length, '0') for t in iterable)


def _expand_ids(ids):
  ids = ids.split(',')
  result = []
  for id in ids:
    if '-' in id:
      begin, end = [int(token) for token in id.split('-')]
      result.extend(_pad_zeros(range(begin, end + 1), len(str(end))))
    else:
      result.append(id)
  return result


# TODO: fix bug!!
def _expand_nodelist(nodelist):
  """
  Updated by haoyuz:
  The following implementation (commented, from original code) does not understand
  patterns like "tiger-i19g7,tiger-i20g[5-7]"
  At this moment we cannot find code that correctly converts the nodelist to hostnames.
  The command `scontrol` does the trick:
    scontrol show hostnames 'compute-b24-[1-3,5-9],compute-b25-[1,4,8]'
  Note that this only works for Python 3.5+ and is NOT backwards compatible.

  # prefix, ids = re.findall("(.*)\[(.*)\]", nodelist)[0]
  # ids = _expand_ids(ids)
  # result = [prefix + str(id) for id in ids]
  # return result

  :param nodelist: List of Slurm node in shortened format
  :return: list of strings, each one is a hostname
  """

  import subprocess
  return subprocess.run(
      ["scontrol show hostname $SLURM_JOB_NODELIST"],
      shell=True,
      stdout=subprocess.PIPE).stdout.decode('utf-8').split()

def _worker_task_id(nodelist, nodename):
  return nodelist.index(nodename)
