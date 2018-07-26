#!/bin/bash
#
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#
#SBATCH --job-name=dist_resnet_cifar10_async
#SBATCH --output=slurm-%x-%j.out
#
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=andrewor@princeton.edu

SLURM_LOG_DIR=/home/andrewor/logs
SCRIPT_PATH=/home/andrewor/models/slurm/dist_resnet_cifar10.sh
TIMESTAMP=`date +%s`

srun --output="$SLURM_LOG_DIR/slurm-%x-%j-%n-$TIMESTAMP.out" "$SCRIPT_PATH" "$TIMESTAMP"

