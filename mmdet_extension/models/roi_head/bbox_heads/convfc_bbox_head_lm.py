# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch

from mmdet.core import multi_apply
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.models.builder import HEADS


@HEADS.register_module()
class Shared2FCBBoxHeadLM(Shared2FCBBoxHead):
    def _get_target_single_lm(
            self, pos_bboxes, pos_gt_bboxes, pos_gt_labels, pos_reg_weight,  # positive
            ig_bboxes, ig_gt_bboxes, ig_gt_labels, ig_reg_weight,  # ignore
            neg_bboxes, cfg):
        num_pos = pos_bboxes.size(0)
        num_ig = ig_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg + num_ig

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        # reliable pseudo labels
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = pos_reg_weight.unsqueeze(1)
        # uncertain pseudo labels
        if num_ig > 0:
            labels[num_pos:num_ig + num_pos] = ig_gt_labels
            label_weights[num_pos:num_ig + num_pos] = ig_reg_weight
            if not self.reg_decoded_bbox:
                ig_bbox_targets = self.bbox_coder.encode(
                    ig_bboxes, ig_gt_bboxes)
            else:
                ig_bbox_targets = ig_gt_bboxes
            bbox_targets[num_pos:num_pos + num_ig, :] = ig_bbox_targets
            bbox_weights[num_pos:num_pos + num_ig, :] = ig_reg_weight.unsqueeze(1)

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets_lm(self,
                       sampling_results,
                       gt_bboxes,
                       gt_labels,
                       rcnn_train_cfg,
                       concat=True
                       ):
        # positive
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_reg_weight = [res.pos_reg_weight for res in sampling_results]
        # ignore
        ig_bboxes_list = [res.ig_bboxes for res in sampling_results]
        ig_gt_bboxes_list = [res.ig_gt_bboxes for res in sampling_results]
        ig_gt_labels_list = [res.ig_gt_labels for res in sampling_results]
        ig_reg_weight = [res.ig_reg_weight for res in sampling_results]
        # negative
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single_lm,
            pos_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, pos_reg_weight,
            ig_bboxes_list, ig_gt_bboxes_list, ig_gt_labels_list, ig_reg_weight,
            neg_bboxes_list, cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
