#!/bin/bash
#
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#
#SBATCH --job-name=dist_resnet_cifar10_sync
#SBATCH --output=slurm-%x-%j.out
#
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=andrewor@princeton.edu

SLURM_LOG_DIR="/home/andrewor/logs"
RUN_PATH="/home/andrewor/models/slurm/run_with_env.sh"
SCRIPT_NAME="dist_resnet_cifar10.sh"

export RESNET_K_SYNC_ENABLED="true"
export RESNET_K_SYNC_STARTING_AGGREGATE_REPLICAS=4
export RESNET_K_SYNC_TOTAL_REPLICAS=4
export RESNET_K_SYNC_SCALING_DURATION=10000

srun --output="$SLURM_LOG_DIR/slurm-%x-%j-%n-$TIMESTAMP.out" "$RUN_PATH" "$SCRIPT_NAME"

