# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Re-implementation: Unbiased teacher for semi-supervised object detection

There are several differences with official implementation:
1. we only use the strong-augmentation version of labeled data rather than \
the strong-augmentation and weak-augmentation version of labeled data.
"""
import numpy as np
import torch

from mmcv.runner.dist_utils import get_dist_info

from mmdet.utils import get_root_logger
from mmdet.models.builder import DETECTORS
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from mmdet_extension.models.detectors import SemiTwoStageDetector


@DETECTORS.register_module()
class UnbiasedTeacher(SemiTwoStageDetector):
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
                 # ut config
                 cfg=dict(),
                 ):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, neck=neck, pretrained=pretrained,
                         ema_config=ema_config, ema_ckpt=ema_ckpt)
        self.debug = cfg.get('debug', False)
        self.num_classes = self.roi_head.bbox_head.num_classes
        self.cur_iter = 0

        # hyper-parameter
        self.score_thr = cfg.get('score_thr', 0.7)
        self.weight_u = cfg.get('weight_u', 2.0)
        self.use_bbox_reg = cfg.get('use_bbox_reg', False)
        self.momentum = cfg.get('momentum', 0.996)

        # analysis
        self.image_num = 0
        self.pseudo_num = np.zeros(self.num_classes)
        self.pseudo_num_tp = np.zeros(self.num_classes)
        self.pseudo_num_gt = np.zeros(self.num_classes)

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1,
    ):
        device = img.device
        self.image_num += len(img_metas_unlabeled)
        self.update_ema_model(self.momentum)
        self.cur_iter += 1
        self.analysis()
        # # ---------------------label data---------------------
        losses = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        losses = self.parse_loss(losses)
        # # -------------------unlabeled data-------------------
        bbox_transform = []
        for img_meta in img_metas_unlabeled_1:
            bbox_transform.append(img_meta.pop('bbox_transform'))
        bbox_results = self.inference_unlabeled(
            img_unlabeled, img_metas_unlabeled, rescale=True
        )
        gt_bboxes_pred, gt_labels_pred = self.create_pseudo_results(
            img_unlabeled_1, bbox_results, bbox_transform, device,
            gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled  # for analysis
        )
        if self.debug:
            self.visual_online(img_unlabeled_1, gt_bboxes_pred, gt_labels_pred)
        losses_unlabeled = self.forward_train(img_unlabeled_1, img_metas_unlabeled_1,
                                              gt_bboxes_pred, gt_labels_pred)
        losses_unlabeled = self.parse_loss(losses_unlabeled)
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            if key.find('bbox') != -1:
                losses_unlabeled[key] = self.weight_u * val if self.use_bbox_reg else 0 * val
            else:
                losses_unlabeled[key] = self.weight_u * val
        losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})
        extra_info = {
            'pseudo_num': torch.Tensor([self.pseudo_num.sum() / self.image_num]).to(device),
            'pseudo_num(acc)': torch.Tensor([self.pseudo_num_tp.sum() / self.pseudo_num.sum()]).to(device)
        }
        losses.update(extra_info)
        return losses

    def create_pseudo_results(self, img, bbox_results, box_transform, device,
                              gt_bboxes=None, gt_labels=None, img_metas=None):
        """using dynamic score to create pseudo results"""
        gt_bboxes_pred, gt_labels_pred = [], []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None
        for b, result in enumerate(bbox_results):
            bboxes, labels = [], []
            if use_gt:
                gt_bbox, gt_label = gt_bboxes[b].cpu().numpy(), gt_labels[b].cpu().numpy()
                scale_factor = img_metas[b]['scale_factor']
                gt_bbox_scale = gt_bbox / scale_factor
            for cls, r in enumerate(result):
                label = cls * np.ones_like(r[:, 0], dtype=np.uint8)
                flag = r[:, -1] >= self.score_thr
                bboxes.append(r[flag][:, :4])
                labels.append(label[flag])
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes[-1]) > 0:
                    overlap = bbox_overlaps(bboxes[-1], gt_bbox_scale[gt_label == cls])
                    iou = overlap.max(-1)
                    self.pseudo_num_tp[cls] += (iou > 0.5).sum()
                self.pseudo_num_gt[cls] += (gt_label == cls).sum()
                self.pseudo_num[cls] += len(bboxes[-1])
            bboxes = np.concatenate(bboxes)
            labels = np.concatenate(labels)
            for bf in box_transform[b]:
                bboxes, labels = bf(bboxes, labels)
            gt_bboxes_pred.append(torch.from_numpy(bboxes).float().to(device))
            gt_labels_pred.append(torch.from_numpy(labels).long().to(device))
        return gt_bboxes_pred, gt_labels_pred

    def analysis(self):
        if self.cur_iter % 500 == 0 and get_dist_info()[0] == 0:
            logger = get_root_logger()
            info = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                             in zip(self.CLASSES, self.pseudo_num, self.pseudo_num_tp)])
            info_gt = ' '.join([f'{a}' for a in self.pseudo_num_gt])
            logger.info(f'pseudo pos: {info}')
            logger.info(f'pseudo gt: {info_gt}')
