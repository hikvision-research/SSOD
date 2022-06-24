# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import random
import numpy as np
import mmcv
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.utils import get_root_logger

from mmdet_extension.core.utils.classes import COCO_CLASSES


@DATASETS.register_module()
class TXTDataset(CustomDataset):
    """support text format dataset
    each line: name h w bbox_num x1 y1 x2 y2 cls ignore ...
    """
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
                 flag_value=0,
                 ignore_label=-2,
                 with_box_id=False,
                 manual_length=None
                 ):
        self.flag_value = flag_value
        self.ignore_label = ignore_label  # ignore = ignore_label + -1*cls
        self.with_box_id = with_box_id  # whether return with box_id (used in instance consistency)
        super().__init__(ann_file, pipeline, classes, data_root, img_prefix, seg_prefix, proposal_file,
                         test_mode, filter_empty_gt)
        self.length = min(manual_length, len(self.data_infos)) if manual_length else len(self.data_infos)

    # override-to avoid error, filter empty image
    def _filter_imgs(self, min_size=32):
        if self.filter_empty_gt:
            valid_inds = []
            for i, line in enumerate(self.data_infos):
                num = int(line.decode().split(' ')[3])
                if num > 0:
                    valid_inds.append(i)
        else:
            valid_inds = list(range(len(self.data_infos)))
        return valid_inds

    # override: not use flag (this may add in the future)
    def _set_group_flag(self):
        self.flag = np.ones(len(self), dtype=np.uint8) * self.flag_value

    def load_annotations(self, ann_file):
        logger = get_root_logger()
        timer = mmcv.Timer()
        ann_list = mmcv.list_from_file(ann_file)
        data_infos = []
        for ann_line in ann_list:
            data_infos.append(ann_line.encode())
        logger.info(f'Loading {len(data_infos)} images, cost {timer.since_start()}')
        return data_infos

    def _parse_str_info(self, str_data):
        line_info = str_data.split()
        height, width = int(line_info[1]), int(line_info[2])
        bbox_number = int(line_info[3])
        bboxes, labels, box_ids = [], [], []
        for idx in range(bbox_number):
            bbox = [float(ann) for ann in line_info[4 + 6 * idx:8 + 6 * idx]]
            if (bbox[3] - bbox[1] < 2.0) or (bbox[2] - bbox[0] < 2.0):
                continue
            if int(line_info[8 + 6 * idx]) > len(self.CLASSES):  # set other cls to background
                continue
            bboxes.append(bbox)
            label, ignore = int(line_info[8 + 6 * idx]) - 1, int(line_info[9 + 6 * idx])
            labels.append(self.ignore_label + label * -1 if ignore else label)
            box_ids.append(idx)
        bboxes = np.array(bboxes).astype(np.float32) if len(bboxes) != 0 else np.empty((0, 4), dtype=np.float32)
        labels = np.array(labels).astype(np.int64) if len(labels) != 0 else np.empty((0,), dtype=np.int64)
        box_ids = np.array(box_ids).astype(np.int64) if len(box_ids) != 0 else np.empty((0,), dtype=np.int64)
        img_info = dict(
            filename=line_info[0].replace('$SPACE', ' '),
            width=width,
            height=height,
        )
        ann_info = dict(
            bboxes=bboxes,
            # it's inconvenience to move box_id as another ann_info (need to re-write pipeline),
            # so we use this format
            labels=np.c_[labels, box_ids] if self.with_box_id else labels,
        )
        return img_info, ann_info

    def _parse_data_info(self, idx):
        line_info = self.data_infos[idx].decode()
        return self._parse_str_info(line_info)

    def get_ann_info(self, idx):
        _, ann_info = self._parse_data_info(idx)
        return ann_info

    def prepare_train_img(self, idx):
        img_info, ann_info = self._parse_data_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx):
        img_info, _ = self._parse_data_info(idx)
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self):
        return self.length if hasattr(self, 'length') else len(self.data_infos)

    def shuffle_data_info(self):
        random.shuffle(self.data_infos)
