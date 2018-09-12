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
    cp "$LOG_DIR/$dir"/*-4-*out .
    if [[ -z `find "$LOG_DIR/$dir" -name "*benchmark*"` ]]; then
      cp "$LOG_DIR/$dir"/*-5-*out .
    fi
    ./make_plots.sh
    mkdir -p "$TAG/$dir"
    tar -czf "$dir.zip" *out
    mv *png "$TAG/$dir"
    mv *zip "$TAG/$dir"
    rm *out
  fi
done

