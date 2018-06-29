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
python3 merge_image_feature.py train_jpg_1 10 ${MOAB_JOBARRAYINDEX}

# Needs reprocess
# d68d17741e6dd2c9151554d94fc06b1124161562fcbb319c4fbeece1b720e60b.h5
# d68d1b2cf8bc3273d0124dcb108c4c71d83d55129be34e7e4ee99ffe828e386f.h5
# d68d18b8708e412be3981c3d00ef77170e824b490229a5463a43cfdeff443eb3.h5

# Submit job
# if [ ${MOAB_JOBARRAYINDEX} -le 10 ]; then
#    python3 merge_image_feature.py train_jpg_0 10 ${MOAB_JOBARRAYINDEX}
# elif [ ${MOAB_JOBARRAYINDEX} -gt 10 -a ${MOAB_JOBARRAYINDEX} -le 20 ]; then
#    python3 merge_image_feature.py train_jpg_1 10 ${MOAB_JOBARRAYINDEX}-10
# elif [ ${MOAB_JOBARRAYINDEX} -gt 20 -a ${MOAB_JOBARRAYINDEX} -le 30 ]; then
#    python3 merge_image_feature.py train_jpg_2 10 ${MOAB_JOBARRAYINDEX}-20
# elif [ ${MOAB_JOBARRAYINDEX} -gt 30 -a ${MOAB_JOBARRAYINDEX} -le 40 ]; then
#    python3 merge_image_feature.py train_jpg_3 10 ${MOAB_JOBARRAYINDEX}-30
# elif [ ${MOAB_JOBARRAYINDEX} -gt 40 ]; then
#    python3 merge_image_feature.py train_jpg_4 10 ${MOAB_JOBARRAYINDEX}-40
# fi