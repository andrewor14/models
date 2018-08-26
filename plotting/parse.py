#!/usr/bin/env python

import datetime
import re
import sys


# Parse data from the log file, return a list of CSVs
def parse_data(log_file):
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
        m1 = re.match(".*learning_rate = ([\.\d]+)", line)
        m2 = re.match(".*cross_entropy = ([\.\d]+)", line)
        m3 = re.match(".*train_accuracy = ([\.\d]+)", line)
        learning_rate = m1.groups()[0]
        cross_entropy = m2.groups()[0]
        train_accuracy = m3.groups()[0]
      elif "global_step/sec" in line:
        m = re.match(".*global_step/sec: ([\.\d]+)", line)
        global_step_per_sec = m.groups()[0]
      elif "loss" in line:
        m = re.match(".*(\d\d\d\d \d\d:\d\d:\d\d).*loss = ([\.\d]+), step = ([\.\d]+)", line)
        (timestamp, loss, step) = m.groups()
        data += [",".join([step, parse_time(timestamp), loss, learning_rate,\
          cross_entropy, train_accuracy, global_step_per_sec or "0"])]
  return data

# Parse timestamp in the format "MMDD hh:mm:ss" into a UNIX timestamp
def parse_time(ts):
  m = re.match("(\d\d)(\d\d) (\d\d):(\d\d):(\d\d)", ts)
  (month, day, hour, minute, second) = m.groups()
  dt = datetime.datetime(year=2018, month=int(month), day=int(day),\
    hour=int(hour), minute=int(minute), second=int(second))
  return dt.strftime('%s')

def main():
  args = sys.argv
  if len(args) <= 1:
    print "Usage: parse.py [log_file]"
    sys.exit(1)
  log_file = args[1]
  out_file = re.sub("\..*$", "", log_file) + ".csv"
  data = parse_data(log_file)
  # Write parsed data into CSV
  with open(out_file, "w") as f:
    f.write("step,timestamp,loss,learning_rate,cross_entropy,train_accuracy,global_step_per_sec\n")
    for d in data:
      f.write(d + "\n")
  print "Wrote to " + out_file

if __name__ == "__main__":
  main()

