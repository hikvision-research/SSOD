# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch
from mmdet.core.bbox.samplers import SamplingResult


class SamplingResultLM(SamplingResult):
    def __init__(self, pos_inds, ig_inds, neg_inds, bboxes, gt_bboxes, gt_bboxes_ignore, assign_result,
                 assign_result_ig, gt_flags):
        self.pos_inds = pos_inds
        self.ig_inds = ig_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.ig_bboxes = bboxes[ig_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]
        self.ig_is_gt = gt_flags[ig_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.num_igs = gt_bboxes_ignore.shape[0]
        self.ig_assigned_gt_inds = assign_result_ig.gt_inds[ig_inds] - 1
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

        if gt_bboxes_ignore.numel() == 0:
            # hack for index error case
            assert self.ig_assigned_gt_inds.numel() == 0
            self.ig_gt_bboxes = torch.empty_like(gt_bboxes_ignore).view(-1, 4)
        else:
            if len(gt_bboxes_ignore.shape) < 2:
                gt_bboxes_ignore = gt_bboxes_ignore.view(-1, 4)
            self.ig_gt_bboxes = gt_bboxes_ignore[self.ig_assigned_gt_inds, :]
        if assign_result_ig.labels is not None:
            self.ig_gt_labels = assign_result_ig.labels[ig_inds]
        else:
            self.ig_gt_labels = None

        # for reliable pseudo label mining
        self.pos_reg_weight = torch.ones_like(self.pos_assigned_gt_inds)
        self.ig_reg_weight = torch.zeros_like(self.ig_assigned_gt_inds)
        self.neg_reg_weight = torch.ones_like(self.neg_bboxes[:, -1])

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.ig_bboxes, self.neg_bboxes])

    @property
    def ignore_flag(self):
        return torch.cat([torch.zeros_like(self.pos_bboxes[:, -1]), torch.ones_like(self.ig_bboxes[:, -1]),
                          torch.zeros_like(self.neg_bboxes[:, -1])]).bool()
