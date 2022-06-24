# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
New COCO Dataset:
1. add manual data-length
"""
import random

from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet_extension.core.utils.classes import COCO_CLASSES


@DATASETS.register_module()
class NewCocoDataset(CocoDataset):
    CLASSES = COCO_CLASSES

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=False,
                 manual_length=None):
        super().__init__(ann_file=ann_file, pipeline=pipeline, classes=classes,
                         data_root=data_root, img_prefix=img_prefix, seg_prefix=seg_prefix,
                         proposal_file=proposal_file, test_mode=test_mode, filter_empty_gt=filter_empty_gt)
        self.length = min(manual_length, len(self.data_infos)) if manual_length else len(self.data_infos)

    def __len__(self):
        return self.length

    def shuffle_data_info(self):
        random.shuffle(self.data_infos)
