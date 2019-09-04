#!/usr/bin/env python3

import os

import tensorflow as tf

from autoscaling.params import *


def log_fn(msg):
  tf.logging.info("[Utility function]: %s" % msg)

def get_price_per_worker_per_second():
  return float(os.getenv(AUTOSCALING_PRICE_PER_WORKER_PER_SECOND, 1/3600))

def get_utility_function(initial_t):
  """
  Return a utility function based on the relevant environment variables.
  """
  name = os.getenv(AUTOSCALING_UTILITY_FUNCTION_NAME)
  args = os.getenv(AUTOSCALING_UTILITY_FUNCTION_ARGS)
  if name is None or args is None:
    return None
  if name == AUTOSCALING_UTILITY_FUNCTION_LINEAR:
    return make_linear_utility_function(args)
  if name == AUTOSCALING_UTILITY_FUNCTION_STEP:
    return make_step_utility_function(args, initial_t)
  raise ValueError("Unknown utility function '%s'" % name)

def make_linear_utility_function(args):
  """
  Return a decreasing linear utility function.
  Args are expected to be a string in the following format: <y-intercept>:<x-intercept>
  """
  y_intercept, x_intercept = args.split(",")
  y_intercept = float(y_intercept)
  x_intercept = float(x_intercept)
  slope = -1 * y_intercept / x_intercept
  def func(t):
    return t * slope + y_intercept
  log_fn("Built linear utility function with y-intercept = %s, x-intercept = %s" %\
    (y_intercept, x_intercept))
  return func

def make_step_utility_function(args, initial_t):
  """
  Return a utility function with decreasing steps.

  Args are expected to be a string in the following format:
    <value1>:<end1>,<value2>:<end2>,<value3>:<end3> ...
  where 'value' refers to the utility and 'end' refers to where this line ends.

  We expect the provided values to be strictly decreasing, and the end points provided
  to be strictly increasing.
  """
  segments = []
  for segment in args.split(","):
    value, end = segment.split(":")
    segments.append((int(value), int(end)))
  # If the first segment did not end at 0, add a point to make it end at 0
  if segments[0][1] != 0:
    segments.insert(0, (segments[0][0], 0))
  # Anything above initial_t will receive negative utility
  if initial_t > segments[-1][1]:
    segments.append((0, initial_t))
  # Bridge the gaps with lines! Just compute the slopes between the points.
  # E.g. (50, 0), (50, 100), (0, 300) => (0, 100, 0, 50), (100, 300, -0.25, 50)
  # E.g. (50, 0), (20, 100), (0, 300) => (0, 100, -0.3, 50), (100, 300, -0.1, 20)
  data = [] # (start point, end point, slope, intercept)
  for i in range(1, len(segments)):
    value1, end1 = segments[i-1]
    value2, end2 = segments[i]
    # Check if segments are strictly decreaseing and end points are strictly increasing
    if value1 < value2:
      raise ValueError("Invalid values in step function args: %s" % args)
    if end1 >= end2:
      raise ValueError("Invalid end points in step function args: %s" % args)
    start_point = end1
    end_point = end2
    slope = (value2 - value1) / (end2 - end1)
    intercept = value1
    data.append((start_point, end_point, slope, intercept))
  log_fn("Built step utility function with the following parameters "\
    "(start, end, slope, intercept): %s" % data)
  # Build the function
  def func(t):
    for start_point, end_point, slope, intercept in data:
      if t < end_point:
        return (t - start_point) * slope + intercept
    # We're going beyond the last end point, so just use the last line,
    # which will probably return negative utility
    start_point, end_point, slope, intercept = data[-1]
    return (t - start_point) * slope + intercept
  return func

