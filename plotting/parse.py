#!/usr/bin/env python

import re
import sys

args = sys.argv
if len(args) <= 1:
  print "Usage: parse.py [log_file]"
  sys.exit(1)
log_file = args[1]
out_file = re.sub("\..*$", "", log_file) + ".csv"

# Parse into list
data = []
with open(log_file, "r") as f:
  loss = None
  learning_rate = None
  cross_entropy = None
  train_accuracy = None
  global_step_per_sec = None
  for line in f.readlines():
    if "tf_logging" not in line:
      continue
    if "learning_rate" in line:
      m = re.match(".*learning_rate = ([\.0-9]+), cross_entropy = ([\.0-9]+), train_accuracy = ([\.0-9]+)", line)
      (learning_rate, cross_entropy, train_accuracy) = m.groups()
    elif "global_step/sec" in line:
      m = re.match(".*global_step/sec: ([\.0-9]+)", line)
      global_step_per_sec = m.groups()[0]
    elif "loss" in line:
      m = re.match(".*loss = ([\.0-9]+), step = ([\.0-9]+)", line)
      (loss, step) = m.groups()
      data += [",".join([step, loss, learning_rate, cross_entropy, train_accuracy, global_step_per_sec or "0"])]

# Write parsed data into CSV
with open(out_file, "w") as f:
  f.write("step,loss,learning_rate,cross_entropy,train_accuracy,global_step_per_sec\n")
  for d in data:
    f.write(d + "\n")

print "Wrote to " + out_file
