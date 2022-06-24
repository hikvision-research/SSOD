# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
semi-supervised two stage detector
"""
import torch

from mmdet.core import bbox2result
from mmdet.models.detectors import TwoStageDetector

from mmdet_extension.models.detectors.semi_base import SemiBaseDetector


class SemiTwoStageDetector(SemiBaseDetector, TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 # ema model
                 ema_config=None,
                 ema_ckpt=None,
                 classes=None
                 ):
        SemiBaseDetector.__init__(self, ema_config=ema_config, ema_ckpt=ema_ckpt, classes=classes)
        TwoStageDetector.__init__(self, backbone=backbone, rpn_head=rpn_head, roi_head=roi_head,
                                  train_cfg=train_cfg, test_cfg=test_cfg, neck=neck, pretrained=pretrained)

    @torch.no_grad()
    def inference_unlabeled(self, img, img_metas, rescale=True, return_feat=False):
        ema_model = self.ema_model.module
        # inference: create pseudo label
        x = ema_model.extract_feat(img)
        proposal_list = ema_model.rpn_head.simple_test_rpn(x, img_metas)
        # bboxes
        det_bboxes, det_labels = ema_model.roi_head.simple_test_bboxes(
            x, img_metas, proposal_list, ema_model.roi_head.test_cfg, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.num_classes)
            for i in range(len(det_bboxes))]
        if return_feat:  # for soft teacher
            return x, bbox_results
        else:
            return bbox_results
