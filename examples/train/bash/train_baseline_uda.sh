#!/usr/bin/env bash

work_dir=$(dirname $0)
cd $work_dir

pip install xonsh

# # ------C2F-------
cd ../
cd xonsh
xonsh train_gpu2.sh ./configs/baseline/baseline_uda.py C2F

# # ------C2B-------
# we use the same baseline with C2F

# # ------K2C-------
#xonsh train_gpu2.sh ./configs/baseline/baseline_uda.py K2C

# # ------S2C-------
#xonsh train_gpu2.sh ./configs/baseline/baseline_uda.py S2C