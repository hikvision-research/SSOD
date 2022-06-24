#!/usr/bin/env bash

work_dir=$(dirname $0)
cd $work_dir

pip install xonsh

cd ../
cd xonsh

# # ====================================
# # fair comparison
# # ====================================
# 1. labelmatch
xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_standard.py 1 1 none # seed percent dataset

# 2. stac
#xonsh train_gpu8.sh ./configs/stac/stac_standard.py 1 1 none # seed percent dataset

# 3. unbiased teacher
#xonsh train_gpu8.sh ./configs/unbiased_teacher/unbiased_teacher_standard.py 1 1 none # seed percent dataset

# 4. soft teacher
#xonsh train_gpu8.sh ./configs/soft_teacher/soft_teacher_standard.py 1 1 none # seed percent dataset


# # ====================================
# # hyper-parameter in paper
# # ====================================
# 1. coco-standard
#xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_standard_paper.py 1 1 none # seed percent dataset

# 2. coco-additional
#xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_additional.py 1 1 none # seed percent dataset

# 3. voc
#xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_voc.py 1 1 none # seed percent dataset