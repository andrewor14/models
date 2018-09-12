#!/bin/bash

if [[ "$#" -ne 1 ]]; then
  echo "No tag provided, exiting."
  exit 1
fi

TAG="$1"

function assign() {
  echo "Assigning $1"
  i=1
  for f in `ls *-"$1"*-0-*`; do
    dir="$TAG/run$i"
    pattern=$(echo "$f" | sed s/-0-.*$//g)
    mkdir -p "$dir"
    mv "$pattern"* "$dir"
    i=$((i+1))
  done
}

assign sync
assign async
assign momentum
assign ksync

