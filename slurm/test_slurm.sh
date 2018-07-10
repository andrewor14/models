#!/bin/bash

echo "Hey guys"
module load cudnn/cuda-8.0/7.0
module load anaconda3/5.0.1
python /home/andrewor/models/samples/core/get_started/premade_estimator.py
echo "Signing off!"

#export PYTHONPATH="$PYTHONPATH:/home/andrewor/models"
#echo $SLURM_JOB_NODELIST
#echo $SLURMD_NODENAME
#date
#sleep 60
#date

