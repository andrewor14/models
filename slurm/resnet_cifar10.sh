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

./setup_env.sh
export PYTHONPATH="$PYTHONPATH:/home/haoyuz/tensorflow-models"
cd /home/haoyuz/tensorflow-models/official/resnet
python cifar10_main.py --data_dir=/tigress/haoyuz/cifar10-dataset \
                       --model_dir=/tigress/haoyuz/resnet_cifar10_model \
                       --num_gpus=4

