#!/usr/bin/env bash

work_dir=$(dirname $0)
cd $work_dir

pip install xonsh

cd ../
cd xonsh

# # ------C2F-------
xonsh train_gpu2.sh ./configs/labelmatch/labelmatch_uda.py C2F

# # ------C2B-------
#xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_uda.py C2B
#xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_uda_prior.py C2B  # with prior from unlabeled data

# # ------K2C-------
#xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_uda.py K2C

# # ------S2C-------
#xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_uda.py S2C