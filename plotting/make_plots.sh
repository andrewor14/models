#!/bin/bash

TITLE="Resnet-50 cifar10, 4 workers, 1 server"
LOG_FILES="$(ls *out | tr '\n' ',' | sed 's/,$//g')"

./plot --x time_elapsed --y train_accuracy --output train_accuracy.png --title "$TITLE" --logs "$LOG_FILES"
./plot --x time_elapsed --y loss --output loss.png --title "$TITLE" --logs "$LOG_FILES"
./plot --x step --y train_accuracy --output train_accuracy_step.png --title "$TITLE" --logs "$LOG_FILES"
./plot --x step --y loss --output loss_step.png --title "$TITLE" --logs "$LOG_FILES"
./plot --x step --y time_elapsed --output step.png --title "$TITLE" --logs "$LOG_FILES"

