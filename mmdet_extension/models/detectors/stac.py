# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Re-implementation: A Simple Semi-Supervised Learning Framework for Object Detection
"""
import torch

from mmdet.models.builder import DETECTORS

from mmdet_extension.models.detectors import SemiTwoStageDetector


@DETECTORS.register_module()
class STAC(SemiTwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 # config
                 cfg=dict(),
                 ):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, neck=neck, pretrained=pretrained)
        self.debug = cfg.get('debug', False)
        self.num_classes = self.roi_head.bbox_head.num_classes

        # hyper-parameter
        self.weight_u = cfg.get('weight_u', 2.0)

        # analysis
        self.image_num = 0
        self.pseudo_num = 0

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled):
        device = img.device
        self.image_num += len(img_metas_unlabeled)
        self.pseudo_num += sum([a.shape[0] for a in gt_labels_unlabeled])
        # # ---------------------label data---------------------
        losses = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        losses = self.parse_loss(losses)
        # # -------------------unlabeled data-------------------
        if self.debug:
            self.visual_online(img_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled)
        losses_unlabeled = self.forward_train(img_unlabeled, img_metas_unlabeled,
                                              gt_bboxes_unlabeled, gt_labels_unlabeled)
        losses_unlabeled = self.parse_loss(losses_unlabeled)
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            losses_unlabeled[key] = self.weight_u * val
        losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})
        extra_info = {
            'pseudo_num': torch.Tensor([self.pseudo_num / self.image_num]).to(device),
        }
        losses.update(extra_info)
        return losses
