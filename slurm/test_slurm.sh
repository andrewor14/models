#!/bin/bash

echo "Hey guys"
./setup_env.sh
python /home/andrewor/models/samples/core/get_started/premade_estimator.py
echo "Signing off!"

#export PYTHONPATH="$PYTHONPATH:/home/andrewor/models"
#echo $SLURM_JOB_NODELIST
#echo $SLURMD_NODENAME
#date
#sleep 60
#date

