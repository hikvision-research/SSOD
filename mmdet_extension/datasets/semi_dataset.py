# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import copy
import random
from torch.utils.data import Dataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import CocoDataset

from mmdet_extension.datasets.txt_style import TXTDataset
from mmdet_extension.core.utils.classes import COCO_CLASSES


@DATASETS.register_module()
class SemiDataset(Dataset):
    CLASSES = COCO_CLASSES

    def __init__(self,
                 ann_file,
                 pipeline,
                 ann_file_u,
                 pipeline_u_share,
                 pipeline_u,
                 pipeline_u_1,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 data_root_u=None,
                 img_prefix_u='',
                 seg_prefix_u=None,
                 proposal_file_u=None,
                 classes=None,
                 filter_empty_gt=True,
                 ):
        super().__init__()

        self.coco_labeled = self.get_data_cls(ann_file)(
            ann_file, pipeline, data_root=data_root, img_prefix=img_prefix,
            seg_prefix=seg_prefix, proposal_file=proposal_file, test_mode=False,
            filter_empty_gt=filter_empty_gt, classes=classes)
        self.coco_unlabeled = self.get_data_cls(ann_file_u)(
            ann_file_u, pipeline_u_share, data_root=data_root_u, img_prefix=img_prefix_u,
            seg_prefix=seg_prefix_u, proposal_file=proposal_file_u, test_mode=False,
            filter_empty_gt=False, classes=classes
        )
        self.CLASSES = self.coco_labeled.get_classes(classes)
        self.pipeline_u = Compose(pipeline_u)
        self.pipeline_u_1 = Compose(pipeline_u_1) if pipeline_u_1 else None

        self.flag = self.coco_unlabeled.flag  # not used

    def get_data_cls(self, ann_file):
        if ann_file.endswith('.json'):
            return CocoDataset
        elif ann_file.endswith('.txt'):
            return TXTDataset
        else:
            raise ValueError(f'please use json or text format annotations')

    def __len__(self):
        return len(self.coco_unlabeled)

    def __getitem__(self, idx):
        idx_label = random.randint(0, len(self.coco_labeled) - 1)
        results = self.coco_labeled[idx_label]

        results_u = self.coco_unlabeled[idx]
        if self.pipeline_u_1:
            results_u_1 = copy.deepcopy(results_u)
            results_u_1 = self.pipeline_u_1(results_u_1)
            results.update({f'{key}_unlabeled_1': val for key, val in results_u_1.items()})
        results_u = self.pipeline_u(results_u)
        results.update({f'{key}_unlabeled': val for key, val in results_u.items()})
        return results

    def update_ann_file(self, ann_file):
        self.coco_unlabeled.data_infos = self.coco_unlabeled.load_annotations(ann_file)
