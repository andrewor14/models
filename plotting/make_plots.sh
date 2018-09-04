#!/bin/bash

TITLE="Resnet-56 cifar10, 4 workers, 1 ps"
LOG_FILES="$(ls *out | tr '\n' ',' | sed 's/,$//g')"
ACCURACY_METRIC="top_1_accuracy"
IS_NEW_FORMAT="$(echo "$LOG_FILES" | grep "benchmark")"

if [[ -z "$IS_NEW_FORMAT" ]]; then
  TITLE="Resnet-50 cifar10, 4 workers, 1 ps"
  ACCURACY_METRIC="train_accuracy"
fi

./plot.py --x time_elapsed --y "$ACCURACY_METRIC" --output "$ACCURACY_METRIC".png --title "$TITLE" --logs "$LOG_FILES"
./plot.py --x time_elapsed --y loss --output loss.png --title "$TITLE" --logs "$LOG_FILES"
./plot.py --x step --y "$ACCURACY_METRIC" --output "$ACCURACY_METRIC"_step.png --title "$TITLE" --logs "$LOG_FILES"
./plot.py --x step --y loss --output loss_step.png --title "$TITLE" --logs "$LOG_FILES"
./plot.py --x step --y time_elapsed_per_step --output step.png --title "$TITLE" --logs "$LOG_FILES"

