#!/bin/bash

MODEL="${MODEL:=Resnet-50-v1.5}"
TITLE="$MODEL imagenet, 4 workers, 1 ps"
LOG_FILES="$(ls *out | tr '\n' ',' | sed 's/,$//g')"
IS_NEW_FORMAT="$(echo "$LOG_FILES" | grep "benchmark")"
EVALUATOR="$(grep "Evaluation" *out)"

if [[ -z "$IS_NEW_FORMAT" ]]; then
  TITLE="Resnet-50 cifar10, 4 workers, 1 ps"
  ./plot.py --x time_elapsed --y train_accuracy --output train_accuracy.png --title "$TITLE" --logs "$LOG_FILES"
  ./plot.py --x step --y train_accuracy --output train_accuracy_step.png --title "$TITLE" --logs "$LOG_FILES"
fi

if [[ -n "$IS_NEW_FORMAT" ]] || [[ -n "$EVALUATOR" ]]; then
  ./plot.py --x time_elapsed --y validation_accuracy --output validation_accuracy.png --title "$TITLE" --logs "$LOG_FILES"
  ./plot.py --x step --y validation_accuracy --output validation_accuracy_step.png --title "$TITLE" --logs "$LOG_FILES"
fi

./plot.py --x time_elapsed --y loss --output loss.png --title "$TITLE" --logs "$LOG_FILES"
./plot.py --x step --y loss --output loss_step.png --title "$TITLE" --logs "$LOG_FILES"
./plot.py --x step --y time_elapsed_per_step --output step.png --title "$TITLE" --logs "$LOG_FILES"

