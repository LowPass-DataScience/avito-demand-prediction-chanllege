#!/bin/bash
#MSUB -A p30072
#MSUB -q short
#MSUB -l walltime=01:00:00
#MSUB -l nodes=1:ppn=28
#MSUB -t img.[1-10]
#PBS -V MOAB_JOBARRAYINDEX

# Setup working directory
source activate kaggle
cd /projects/p30072/kaggle/Kaggle-DataScience-Projects/avito-demand-prediction-challenge/python_codes/

# Submit job
python3 process_image.py test_jpg 10 ${MOAB_JOBARRAYINDEX}