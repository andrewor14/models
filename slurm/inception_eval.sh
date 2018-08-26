#!/bin/bash
# parallel job using 4 GPU and runs for 48 hours (max) 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH -t 48:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=haoyuz@princeton.edu

cd /home/haoyuz/tensorflow-models/research/inception
bazel-bin/inception/imagenet_eval --num_gpus=4 --checkpoint_dir=/tigress/haoyuz/imagenet_train --data_dir=/tigress/haoyuz/imagenet-dataset --eval_dir=/tigress/haoyuz/imagenet_eval

