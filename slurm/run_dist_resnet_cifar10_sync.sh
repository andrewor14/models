#!/bin/bash
#
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#
#SBATCH --job-name=dist_resnet_cifar10_sync
#SBATCH --output=slurm-%x-%j.out
#
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=andrewor@princeton.edu

SLURM_LOG_DIR=/home/andrewor/logs
SCRIPT_PATH=/home/andrewor/models/slurm/dist_resnet_cifar10.sh
TIMESTAMP=`date +%s`

export ANDREW_RESNET_SYNC_ENABLED="true"
export ANDREW_RESNET_SYNC_AGGREGATE_REPLICAS=2
export ANDREW_RESNET_SYNC_TOTAL_REPLICAS=2

srun --output="$SLURM_LOG_DIR/slurm-%x-%j-%n-$TIMESTAMP.out" "$SCRIPT_PATH" "$TIMESTAMP"

