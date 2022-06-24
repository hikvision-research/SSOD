# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
standard roi_head for Soft-Teacher
"""
from mmdet.models.builder import HEADS
from mmdet.core import bbox2roi

from mmdet_extension.models.roi_head import StandardRoIHeadBase


@HEADS.register_module()
class StandardRoIHeadST(StandardRoIHeadBase):
    def forward_train_step1(self,
                            x,
                            img_metas,
                            proposal_list,
                            gt_bboxes,
                            gt_labels,
                            ):
        num_imgs = len(img_metas)
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], None, gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result, proposal_list[i],
                gt_bboxes[i], gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        return sampling_results

    def forward_train_step2(self,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            soft_weight,
                            ):
        losses = dict()
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        # reset the negative label weight
        bbox_targets = list(bbox_targets)
        bbox_targets[1] = soft_weight
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        losses.update(bbox_results['loss_bbox'])
        return losses