# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch
from mmdet.core.bbox.assigners import AssignResult


class AssignResultLM(AssignResult):
    def add_ig_(self, gt_labels):
        # assign as -1 for ignore
        self_inds = -1 * torch.ones(len(gt_labels), dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
