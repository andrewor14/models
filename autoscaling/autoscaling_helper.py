#!/usr/bin/env python3

import os
import traceback

from autoscaling.agent import AutoscalingAgent
from autoscaling.params import *
from autoscaling.callback import AutoscalingCallback
from autoscaling.schedule_callback import PeriodicSpawnScheduleCallback
from deploy import slurm_helper


def run_keras(flags_obj, do_run):
  """ 
  Wrapper around main loop that handles changes in cluster membership.

  The real computation logic is specified through `do_run`, a function that takes in
  two arguments, `flags_obj` and an `AutoscalingCallback`.
  """
  # If TF_CONFIG is not provided, set it based on environment variables from slurm or MPI
  if "TF_CONFIG" not in os.environ:
    if slurm_helper.running_through_slurm():
      num_ps = int(os.getenv("NUM_PARAMETER_SERVERS", "1"))
      slurm_helper.set_tf_config(num_ps)
      # TODO: divide CUDA_VISIBLE_DEVICES across workers sharing the same machine for slurm too
    elif flags_obj.use_horovod:
      from deploy import mpi_helper
      mpi_helper.set_tf_config()
      if flags_obj.num_gpus > 0:
        mpi_helper.set_cuda_visible_devices(flags_obj.num_gpus)

  # Keep track of cluster membership changes through an autoscaling hook
  autoscaling_agent = AutoscalingAgent()
  autoscaling_callback = AutoscalingCallback(autoscaling_agent)

  while autoscaling_agent.status != AutoscalingStatus.TERMINATED:
    try:
      autoscaling_agent.initialize()
      result = do_run(flags_obj, autoscaling_callback)
      autoscaling_agent.on_restart()
      autoscaling_callback.reset()
    except Exception as e:
      tf.compat.v1.logging.error("Exception in resnet_main: %s (%s)" %\
        (e, e.__class__.__name__))
      traceback.print_exc()
      raise e
  return result

