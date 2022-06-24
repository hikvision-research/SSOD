# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Augmentation in SSOD
"""
from mmdet.datasets.pipelines import Albu
from mmdet.datasets import PIPELINES

from mmdet_extension.datasets.pipelines.transforms_box import RandomErasing, RandomErasingBox


# # -------------------------Unbiased Teacher augmentation-------------------------
class RandomErase(object):
    def __init__(self, use_box=False):
        CLS = RandomErasingBox if use_box else RandomErasing
        self.transforms = [
            CLS(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
            CLS(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
            CLS(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random")
        ]

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
        return results


class AugmentationUTWeak(object):
    def __init__(self):
        self.transforms_1 = Albu(transforms=[
            dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            dict(type='ToGray', p=0.2),
            dict(type='GaussianBlur', sigma_limit=(0.1, 2.0), p=0.2),
        ], bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_labels']),
            keymap={'img': 'image', 'gt_bboxes': 'bboxes'}
        )

    def __call__(self, results):
        results = self.transforms_1(results)
        return results


class AugmentationUTStrong(object):
    def __init__(self, use_re=True, use_box=False):
        self.transforms_1 = Albu(transforms=[
            dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            dict(type='ToGray', p=0.2),
            dict(type='GaussianBlur', sigma_limit=(0.1, 2.0), p=0.5),
        ], bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_labels']),
            keymap={'img': 'image', 'gt_bboxes': 'bboxes'}
        )
        self.transforms_2 = RandomErase(use_box)
        self.use_re = use_re

    def __call__(self, results):
        results = self.transforms_1(results)
        if self.use_re:
            results = self.transforms_2(results)
        return results


@PIPELINES.register_module()
class AugmentationUT(object):
    def __init__(self, use_weak=False, use_re=True, use_box=False):
        if use_weak:
            self.transforms = AugmentationUTWeak()
        else:
            self.transforms = AugmentationUTStrong(use_re=use_re, use_box=use_box)

    def __call__(self, results):
        results = self.transforms(results)
        return results
