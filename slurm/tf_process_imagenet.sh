#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=haoyuz@princeton.edu

export DATA_DIR=/tigress/haoyuz/imagenet-dataset
cd /home/haoyuz/tensorflow-models/research/inception
bazel-bin/inception/tf_preprocess_imagenet "${DATA_DIR}"

