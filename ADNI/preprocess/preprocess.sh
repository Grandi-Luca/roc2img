#!/bin/bash

DATA_PATH="/mnt/shared_nfs/brunofolder/MERGE/WALTER/IMGS/a1"
REFERENCE_ATLAS="/path/to/reference/atlas"
AXIAL_SIZE=90
RE_PROCESS=False
SAVE_2D=False

python3 adni_preprocessing.py --data_path $DATA_PATH --re_process $RE_PROCESS --axial_size $AXIAL_SIZE --save_2d $SAVE_2D --reference_atlas_location $REFERENCE_ATLAS