# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection

"""
standard roi_head for LabelMatch
"""
import torch

from mmdet.models.builder import HEADS
from mmdet.core import bbox2roi
from mmdet.models.roi_heads import StandardRoIHead


@HEADS.register_module()
class StandardRoIHeadBase(StandardRoIHead):
    def simple_test_bboxes_base(self, x, img_metas, proposals):
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # the bbox prediction of some detectors like SABL is not Tensor
        if isinstance(bbox_pred, torch.Tensor):
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        else:
            bbox_pred = self.bbox_head.bbox_pred_split(
                bbox_pred, num_proposals_per_img)
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            bboxes = self.bbox_head.bbox_coder.decode(
                rois[i][:, 1:], bbox_pred[i], max_shape=img_shapes[i])
            det_bboxes.append(bboxes)
            det_labels.append(cls_score[i])
        return det_bboxes, det_labels
