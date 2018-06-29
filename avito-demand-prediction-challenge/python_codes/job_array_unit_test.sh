#!/bin/bash
#MSUB -A    
#MSUB -q short
#MSUB -l walltime=00:05:00
#MSUB -l nodes=1:ppn=28
#MSUB -t img.[1-2]
#PBS -V MOAB_JOBARRAYINDEX

source activate kaggle
cd /projects/p30072/kaggle/Kaggle-DataScience-Projects/avito-demand-prediction-challenge/python_codes/

python3 ./jobArrayUnitTest.py ${MOAB_JOBARRAYINDEX}