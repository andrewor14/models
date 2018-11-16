#!/bin/bash

LOG_DIR="/Users/andrew/Documents/dev/tf-logs"
TAG="$1"

if [[ -z "$TAG" ]]; then
  echo "No tag provided. Exiting."
  exit 1
fi

LOG_DIR="$LOG_DIR/$TAG"
for dir in `ls "$LOG_DIR"`; do
  if [[ -d "$LOG_DIR/$dir" ]] && [[ "$dir" == run* ]]; then
    echo "Processing logs in $LOG_DIR/$dir"
    # Assume the second machine is a worker
    if [[ -n `find "$LOG_DIR/$dir" -name "*-1-*out"` ]]; then
      cp "$LOG_DIR/$dir"/*-1-*out .
      # If this is the models repo also try to get the evaluator
      if [[ -z `find "$LOG_DIR/$dir" -name "*benchmark*"` ]]; then
        cp "$LOG_DIR/$dir"/*-5-*out .
      fi
    else
      # This is local mode, so just grab the log
      TOCOPY="$(grep images $LOG_DIR/$dir/*-0-*out | awk -F ":" '{print $1}' | uniq)"
      cp "$TOCOPY" .
    fi
    ./make_plots.sh
    mkdir -p "$TAG/$dir"
    tar -czf "$dir.zip" *out
    mv *png "$TAG/$dir"
    mv *zip "$TAG/$dir"
    rm *out
  fi
done

