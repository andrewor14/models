#!/usr/bin/env python

import sys

args = sys.argv
if len(args) != 2:
  print "Usage: ./find_run_time.py [log_path]"
  sys.exit(1)

log_path = args[1]

# First, find a keyword to grep for (e.g. I11, for something run in November)
keyword = None
with open(log_path) as f:
  for line in f.readlines():
    if "tf_logging" in line and "Params:" in line:
      keyword = line.split()[0]
      if not keyword.startswith("I"):
        raise ValueError("Expected keyword to start with 'I', was '%s'" % keyword)
      # Grab 'I' and the month, which we assume is encoded with 2 digits
      keyword = keyword[:3]
      break
print "Grepping for keyword '%s'" % keyword

# Second, use the keyword to filter out lines with timestamps
first_timestamp = ""
last_timestamp = ""
with open(log_path) as f:
  timestamp_lines = [line for line in f.readlines() if line.startswith(keyword)]
  first_timestamp = timestamp_lines[0].split()[1]
  last_timestamp = timestamp_lines[-1].split()[1]
print "First timestamp = '%s'" % first_timestamp
print "Last timestamp = '%s'" % last_timestamp

# Parse a timestamp into its parts, e.g. 07:04:38.495621 -> (7, 4, 38)
def parse_timestamp(timestamp):
  (hours, minutes, seconds) = timestamp.split(":")
  hours = int(hours)
  minutes = int(minutes)
  seconds = int(seconds.split(".")[0])
  return (hours, minutes, seconds)

# Compute the difference between two timestamps, return answer in seconds
# This assumes the second timestamp comes later
def calculate_difference(timestamp1, timestamp2):
  (hours1, minutes1, seconds1) = parse_timestamp(timestamp1)
  (hours2, minutes2, seconds2) = parse_timestamp(timestamp2)
  difference = (hours2 - hours1) * 3600 + (minutes2 - minutes1) * 60 + (seconds2 - seconds1)
  # Account for when the experiment runs across the day boundary
  if difference < 0:
    difference += 3600 * 24
  return difference

# Finally, parse the timestamps and compute the difference
time_elapsed_seconds = calculate_difference(first_timestamp, last_timestamp)
print "Run time: %s seconds (%s hours)" % (time_elapsed_seconds, time_elapsed_seconds / float(3600))

