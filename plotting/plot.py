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

# Make log file name more human-readable
def format_name(log_file):
  name = log_file.lstrip("slurm-")
  name = re.sub("\..*$", "", name)
  name = re.sub("[-0-9]+$", "", name)
  return name

# Return the values for a label, printing all known labels if the one requested is unknown
def get_values(label, data, known_labels):
  if label not in data:
    print "Unknown label '%s'. Choose from %s." % (label, known_labels)
    sys.exit(1)
  return data[label]

# Parse and plot data from the specified log file
def plot_data(x_label, y_label, log_file, ax):
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
      if label == "timestamp":
        label = "time_elapsed"
      data[label] = []
      labels.append(label)
    # Parse data
    for line in lines[1:]:
      split = line.strip().split(",")
      if len(split) != len(labels):
        raise Exception("Number of values (%s) does not match number of columns in header (%s)"\
          % (len(split), len(labels)))
      for i, value in enumerate(split):
        label = labels[i]
        # Format value according to label
        if label == "step":
          value = int(value)
        elif label == "time_elapsed":
          value = long(value)
          if first_timestamp is None:
            first_timestamp = value
          value -= first_timestamp
        else:
          value = float(value)
        data[label].append(value)
  # Plot the requested labels
  x_data = get_values(x_label, data, labels)
  y_data = get_values(y_label, data, labels)
  name = format_name(log_file)
  if y_label == "global_step_per_sec":
    x_data = x_data[1:]
    y_data = y_data[1:]
  ax.plot(x_data, y_data, "-x", label=name)
  # Clean up
  os.remove(csv_file)

def main():
  # Parse arguments, e.g.
  # ./plot.py --x time_elapsed --y loss --output loss.png
  #   --title "Resnet50 cifar10, 2 workers (4 GPUs), 1 server"
  #   --logs slurm-dist_resnet_cifar10-async-1053120-1-1532482561.out,slurm-dist_resnet_cifar10-sync-1053119-1-1532445251.out
  parser = argparse.ArgumentParser()
  parser.add_argument("--x", help="x label", default="time_elapsed")
  parser.add_argument("--y", help="y label", default="train_accuracy")
  parser.add_argument("--title", help="plot title")
  parser.add_argument("--logs", help="comma separated path(s) to one or more log files", required=True)
  parser.add_argument("--output", help="path to output file")
  args = parser.parse_args()
  x_label = args.x
  y_label = args.y
  title = args.title
  log_files = args.logs.split(",")
  out_file = args.output or y_label + ".png"

  # Plot it
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  for log_file in log_files:
    plot_data(x_label, y_label, log_file, ax)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  legend = ax.legend(loc="best")
  if title is not None:
    ax.set_title(title)
  fig.savefig(out_file, bbox_extra_artists=(legend,), bbox_inches='tight')
  print "Wrote to " + out_file

if __name__ == "__main__":
  main()

