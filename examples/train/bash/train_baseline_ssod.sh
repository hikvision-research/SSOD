#!/usr/bin/env bash

work_dir=$(dirname $0)
cd $work_dir

pip install xonsh

cd ../
cd xonsh
xonsh train_gpu2.sh ./configs/baseline/baseline_ssod.py 1 1 coco-standard # seed percent dataset

# # -----------voc----------
# xonsh train_gpu2.sh ./configs/baseline/baseline_ssod.py 1 1 voc

# # -----------coco-additional----------
# xonsh train_gpu8.sh ./configs/baseline/baseline_ssod.py 1 1 coco-additional