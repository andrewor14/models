#!/usr/bin/env python

import datetime
import re
import sys


# Parse data from the log file, return a list of CSVs
def parse_data(log_file):
  data = []
  use_new_format = False
  evaluator = False
  with open(log_file, "r") as f:
    loss = None
    learning_rate = None
    cross_entropy = None
    train_accuracy = None
    global_step_per_sec = None
    for line in f.readlines():
      if "tf_logging" not in line:
        continue
      if "Img/sec" in line:
        use_new_format = True
        continue
      if "images/sec" in line and "jitter" in line:
        use_new_format = True
        m1 = re.match(".*(\d\d\d\d \d\d:\d\d:\d\d\.[\d]+).*\s+([\.\d]+)\s+images/sec: ([\.\d]+)", line)
        m2 = re.match(".*\s+([\.\d]+)\s+([\.\d]+)\s+([\.\d]+)", line)
        (timestamp, step, images_per_sec) = m1.groups()
        (loss, top_1_accuracy, top_5_accuracy) = m2.groups()
        data += [",".join([step, parse_time(timestamp), loss, top_1_accuracy, top_5_accuracy])]
      if "Evaluation" in line:
        use_new_format = False
        evaluator = True
        continue
      if not use_new_format:
        if evaluator:
          if "Saving dict for global step" in line:
            m = re.match(".*(\d\d\d\d \d\d:\d\d:\d\d\.[\d]+).*accuracy = ([\.\d]+), global_step = ([\.\d]+), loss = ([\.\d]+)", line)
            (timestamp, accuracy, step, loss) = m.groups()
            data += [",".join([step, parse_time(timestamp), loss, accuracy])]
        elif "learning_rate" in line:
          m1 = re.match(".*learning_rate = ([\.\d]+)", line)
          m2 = re.match(".*cross_entropy = ([\.\d]+)", line)
          m3 = re.match(".*train_accuracy = ([\.\d]+)", line)
          learning_rate = m1.groups()[0]
          cross_entropy = m2.groups()[0]
          train_accuracy = m3.groups()[0]
        elif "global_step/sec" in line:
          m = re.match(".*global_step/sec: ([\.\d]+)", line)
          global_step_per_sec = m.groups()[0]
        elif "loss = " in line:
          m = re.match(".*(\d\d\d\d \d\d:\d\d:\d\d\.[\d]+).*loss = ([\.\d]+), step = ([\.\d]+)", line)
          (timestamp, loss, step) = m.groups()
          data += [",".join([step, parse_time(timestamp), loss, learning_rate,\
            cross_entropy, train_accuracy, global_step_per_sec or "0"])]
  # Insert header based on the format
  header = None
  if use_new_format:
    header = "step,timestamp,loss,top_1_accuracy,top_5_accuracy"
  elif evaluator:
    header = "step,timestamp,loss,top_1_accuracy"
  else:
    header = "step,timestamp,loss,learning_rate,cross_entropy,train_accuracy,global_step_per_sec"
  data.insert(0, header)
  return data

# Parse timestamp in the format "MMDD hh:mm:ss.[...ms...]" into a UNIX timestamp (unit = ms)
def parse_time(ts):
  m = re.match("(\d\d)(\d\d) (\d\d):(\d\d):(\d\d)(.*)", ts)
  (month, day, hour, minute, second, milliseconds) = m.groups()
  milliseconds = long(float(milliseconds) * 1000)
  dt = datetime.datetime(year=2018, month=int(month), day=int(day),\
    hour=int(hour), minute=int(minute), second=int(second))
  return str(long(dt.strftime('%s')) * 1000 + milliseconds)

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
    for d in data:
      f.write(d + "\n")
  print "Wrote to " + out_file

if __name__ == "__main__":
  main()

