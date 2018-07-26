#!/usr/bin/env python

import matplotlib.pyplot as plt 
import math
import numpy as np
import os
import re
import scipy.stats as stats
from subprocess import Popen, PIPE
import sys 


# Parse and plot data from each file
def plot_data(ax, log_files):
  for log_file in log_files:
    Popen(['./parse.py', log_file]).communicate()
    csv_file = re.sub("\..*$", "", log_file) + ".csv"
    step, time_elapsed, loss, learning_rate, cross_entropy, train_accuracy, global_step_per_sec =\
      [], [], [], [], [], [], []
    first_timestamp = None
    with open(csv_file, "r") as f:
      lines = f.readlines()[1:] # skip header
      for line in lines:
        s = line.split(",")
        step.append(int(s[0]))
        this_timestamp = long(s[1])
        if first_timestamp is None:
          first_timestamp = this_timestamp
        time_elapsed.append(this_timestamp - first_timestamp)
        loss.append(float(s[2]))
        learning_rate.append(float(s[3]))
        cross_entropy.append(float(s[4]))
        train_accuracy.append(float(s[5]))
        global_step_per_sec.append(float(s[6]))
    name = log_file.lstrip("slurm-")
    name = re.sub("\..*$", "", name)
    name = re.sub("[-0-9]+$", "", name)
    # Pick and choose what you want to plot!
    #ax.plot(step, learning_rate, "-x", label="learning_rate (%s)" % name)
    #ax.plot(step, cross_entropy, "-x", label="cross_entropy (%s)" % name)
    #ax.plot(step[1:], global_step_per_sec[1:], "-x", label="global_step_per_sec (%s)" % name)
    ax.plot(time_elapsed, loss, "-x", label="loss (%s)" % name)
    #ax.plot(time_elapsed, train_accuracy, "-x", label="train accuracy (%s)" % name)
    os.remove(csv_file)

def main():
  args = sys.argv
  if len(args) <= 2:
    print "Usage: plot.py [title] [log_file1] [log_file2] ..."
    sys.exit(1)
  title = args[1]
  log_files = args[2:]
  out_file = "output.png"
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  plot_data(ax, log_files)
  ax.set_xlabel("time elapsed (s)")
  ax.set_ylabel("value")
  legend = ax.legend(loc="upper center")
  ax.set_title(title)
  fig.savefig(out_file, bbox_extra_artists=(legend,), bbox_inches='tight')
  print "Wrote to " + out_file

if __name__ == "__main__":
  main()

