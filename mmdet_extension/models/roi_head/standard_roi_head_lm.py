# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection

"""
standard roi_head for LabelMatch
"""
import torch

from mmdet.models.builder import HEADS
from mmdet.core import bbox2roi

from mmdet_extension.models.roi_head import StandardRoIHeadBase


@HEADS.register_module()
class StandardRoIHeadLM(StandardRoIHeadBase):
    def forward_train_step1(self,
                            x,
                            img_metas,
                            proposal_list,
                            gt_bboxes,
                            gt_labels,
                            gt_bboxes_ignore=None,
                            gt_labels_ignore=None,
                            ):
        num_imgs = len(img_metas)
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], None, gt_labels[i])
            assign_result_ig = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes_ignore[i], None, gt_labels_ignore[i])
            sampling_result = self.bbox_sampler.sample_pos_ig(
                assign_result, assign_result_ig, proposal_list[i],
                gt_bboxes[i], gt_labels[i], gt_bboxes_ignore[i], gt_labels_ignore[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        return sampling_results

    def forward_train_step2(self,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels
                            ):
        losses = dict()
        rois = bbox2roi([res.bboxes for res in sampling_results])
        flag = torch.cat([res.ignore_flag for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets_lm(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        losses.update(bbox_results['loss_bbox'])
        scores = bbox_results['cls_score'][flag]
        return losses, scores
