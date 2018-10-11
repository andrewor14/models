#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt 
import math
import numpy as np
import os
import re
import scipy.stats as stats
from subprocess import Popen, PIPE
import sys 

# Some special labels
STEP = "step"
TIMESTAMP = "timestamp"
TIME_ELAPSED = "time_elapsed"
TIME_ELAPSED_PER_STEP = "time_elapsed_per_step"
TOP_1_ACCURACY = "top_1_accuracy"
VALIDATION_ACCURACY = "validation_accuracy"
GLOBAL_STEP_PER_SEC = "global_step_per_sec"

# Number of values to take the average over to calculate the converged value
CONVERGED_AVERAGE_OVER = 20

# Make log file name more human-readable
def format_name(log_file):
  name = log_file.lstrip("slurm-")
  name = re.sub("\..*$", "", name)
  name = re.sub("-[-0-9]+$", "", name)
  name = re.sub("momentum-false", "async", name)
  name = re.sub("momentum-true", "sync2", name)
  return name

# Return the values for a label, printing all known labels if the one requested is unknown
def get_values(label, data, known_labels):
  if label not in data:
    print "Unknown label '%s'. Choose from %s." % (label, known_labels)
    sys.exit(1)
  return data[label]

# Return the label text for a certain axis on the plot
def get_label_text(label, time_unit):
  text = label
  if label == TIME_ELAPSED or label == TIME_ELAPSED_PER_STEP:
    text = "%s (%s)" % (text, time_unit)
  return text.replace("_", " ")

# Return whether the given log file describes an evaluator in the old format
def is_evaluator(log_file):
  with open(log_file, "r") as f:
    for line in f.readlines():
      if "Evaluation" in line:
        return True
  return False

# Return whether the given log file uses the new format
def is_new_format(log_file):
  return "benchmark" in log_file

# Parse data and labels from the specified log file
def parse_data(log_file):
  # For now, we only support validation_accuracy for evaluator logs
  if not is_new_format(log_file):
    evaluator = is_evaluator(log_file)
    if evaluator and y_label != VALIDATION_ACCURACY:
      return
    if not evaluator and y_label == VALIDATION_ACCURACY:
      return
  Popen(['./parse.py', log_file], stdout=PIPE, stderr=PIPE).communicate()
  csv_file = re.sub("\..*$", "", log_file) + ".csv"
  labels = []
  data = {} # label -> list of values
  first_timestamp = None
  # Parse CSV file for values
  with open(csv_file, "r") as f:
    lines = f.readlines()
    # Parse labels
    for label in lines[0].strip().split(","):
      if label == TIMESTAMP:
        label = TIME_ELAPSED
      if label == TOP_1_ACCURACY:
        label = VALIDATION_ACCURACY
      data[label] = []
      labels.append(label)
    # Add custom label time_elapsed_per_step
    data[TIME_ELAPSED_PER_STEP] = []
    labels.append(TIME_ELAPSED_PER_STEP)
    # Parse data
    for line in lines[1:]:
      split = line.strip().split(",")
      num_values = len(split)
      num_columns = len(labels) - 1 # don't count time_elapsed_per_step
      if num_values != num_columns:
        raise Exception("Number of values (%s) does not match number of columns in header (%s)"\
          % (num_values, num_columns))
      for i, value in enumerate(split):
        label = labels[i]
        # Format value according to label
        if label == STEP:
          value = int(value)
        elif label == TIME_ELAPSED:
          value = long(value)
          if first_timestamp is None:
            first_timestamp = value
          value -= first_timestamp
        else:
          value = float(value)
        data[label].append(value)
      # Record time elapsed per step
      if len(data[TIME_ELAPSED]) > 0 and len(data[STEP]) > 0:
        time_elapsed_start = data[TIME_ELAPSED][-2] if len(data[TIME_ELAPSED]) > 1 else 0
        time_elapsed_delta = data[TIME_ELAPSED][-1] - time_elapsed_start
        step_start = data[STEP][-2] if len(data[STEP]) > 1 else 0
        step_delta = data[STEP][-1] - step_start
        if step_delta > 0:
          data[TIME_ELAPSED_PER_STEP].append(float(time_elapsed_delta) / step_delta)
  # Clean up
  os.remove(csv_file)
  return (data, labels)

# Plot data given the specified labels
def plot_data(x_label, y_label, data, labels, log_file, ax):
  # Plot the requested labels
  x_data = get_values(x_label, data, labels)
  y_data = get_values(y_label, data, labels)
  name = format_name(log_file)
  if y_label == GLOBAL_STEP_PER_SEC:
    x_data = x_data[1:]
    y_data = y_data[1:]
  if y_label == TIME_ELAPSED_PER_STEP:
    x_data = x_data[len(x_data) - len(y_data):]
  # Append converged value to label
  last_n = y_data[-CONVERGED_AVERAGE_OVER:]
  label = name
  if len(last_n) > 0:
    converged_value = sum(last_n) / len(last_n)
    label += " (%.3f)" % converged_value
  # Pick a style
  color = None
  linewidth = 1
  fmt = "-"
  #if "async2" in log_file:
  #  color = "magenta"
  #elif "async" in log_file:
  #  color = "red"
  #elif "ksync" in log_file:
  #  color = "green"
  #  linewidth = 3
  #elif "sync" in log_file:
  #  color = "blue"
  #else:
  #  color = "black"
  ax.plot(x_data, y_data, fmt, label=label, linewidth=linewidth)

def main():
  # Parse arguments, e.g.
  # ./plot.py --x time_elapsed --y loss --output loss.png
  #   --title "Resnet50 cifar10, 2 workers (4 GPUs), 1 parameter server"
  #   --logs slurm-dist_resnet_cifar10-async-1053120-1-1532482561.out,slurm-dist_resnet_cifar10-sync-1053119-1-1532445251.out
  parser = argparse.ArgumentParser()
  parser.add_argument("--x", help="x label", default=TIME_ELAPSED)
  parser.add_argument("--y", help="y label", default="validation_accuracy")
  parser.add_argument("--title", help="plot title")
  parser.add_argument("--logs", help="comma separated path(s) to one or more log files", required=True)
  parser.add_argument("--output", help="path to output file")
  args = parser.parse_args()
  x_label = args.x
  y_label = args.y
  title = args.title
  log_files = args.logs.split(",")
  out_file = args.output or y_label + ".png"

  # Parse data from all log files
  # For now, assume labels are the same across all log files
  labels = None
  all_data = []
  for log_file in log_files:
    (d, l) = parse_data(log_file)
    all_data.append(d)
    labels = l

  # Figure out units for time labels, converting the corresponding data if necessary
  # For now, assume you can't specify time labels on both axes
  our_time_label = None
  our_time_unit = "ms"
  time_labels = [TIME_ELAPSED, TIME_ELAPSED_PER_STEP]
  for tl in time_labels:
    if x_label == tl or y_label == tl:
      our_time_label = tl
      break
  if our_time_label is not None:
    # Find the max value for the time label across all log files
    multiplier = 1
    max_value = max([max(get_values(our_time_label, d, labels)) for d in all_data])
    if max_value > 100 * 1000 * 1000:
      our_time_unit = "h"
      multiplier = 1000 * 60 * 60
    elif max_value > 100 * 1000:
      our_time_unit = "s"
      multiplier = 1000
    for d in all_data:
      d[our_time_label] = [float(t) / multiplier for t in get_values(our_time_label, d, labels)]

  # Plot it
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  for i, log_file in enumerate(log_files):
    plot_data(x_label, y_label, all_data[i], labels, log_file, ax)
  x_label_text = get_label_text(x_label, our_time_unit)
  y_label_text = get_label_text(y_label, our_time_unit)
  ax.set_xlabel(x_label_text)
  ax.set_ylabel(y_label_text)
  legend = ax.legend(loc="best")
  if title is not None:
    ax.set_title(title)
  fig.savefig(out_file, bbox_extra_artists=(legend,), bbox_inches='tight')
  print "Wrote to " + out_file

if __name__ == "__main__":
  main()

