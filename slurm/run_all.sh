#!/bin/bash

sbatch run_dist_resnet_cifar10_sync.sh
sbatch run_dist_resnet_cifar10_async.sh
sbatch run_dist_resnet_cifar10_ksync.sh

