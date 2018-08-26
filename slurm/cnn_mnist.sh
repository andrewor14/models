#!/bin/bash
# parallel job using 1 GPU and runs for 4 hours (max) 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=haoyuz@princeton.edu

cd /home/haoyuz
python models/cnn_mnist.py

