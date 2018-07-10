#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1g
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#
#SBATCH --job-name=test_slurm
#SBATCH --output=slurm-%x-%j.out
#
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=andrewor@princeton.edu

SLURM_LOG_DIR=/home/andrewor/logs
SCRIPT_PATH=/home/andrewor/models/slurm/test_slurm.sh

srun --output=$SLURM_LOG_DIR/slurm-%x-%j-%n.out $SCRIPT_PATH

