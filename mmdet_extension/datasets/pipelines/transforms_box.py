# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
augmentation with "box transform": convert pseudo labels from weak to strong
"""
import random
import numpy as np

import mmcv
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import RandomFlip, Resize
from mmdet_extension.datasets.pipelines.transforms import RandomErasing


# support bbox transform
@PIPELINES.register_module()
class AddBBoxTransform(object):
    def __call__(self, results):
        results['bbox_transform'] = []
        return results


@PIPELINES.register_module()
class ResizeBox(Resize):
    class BboxResize(object):
        def __init__(self, img_shape, scale_factor, bbox_clip_border, scale=None, keep_ratio=True):
            self.img_shape = img_shape
            self.scale = scale
            self.scale_factor = scale_factor
            self.keep_ratio = keep_ratio
            self.bbox_clip_border = bbox_clip_border

        def __call__(self, bboxes, labels, masks=None):
            bboxes = bboxes * self.scale_factor
            if self.bbox_clip_border:
                img_shape = self.img_shape
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            if masks is None:
                return bboxes, labels
            if self.keep_ratio:
                masks = masks.rescale(self.scale, interpolation='bilinear')
            else:
                masks = masks.resize(self.img_shape[:2], interpolation='bilinear')
            return bboxes, labels, masks

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        results['bbox_transform'].append(self.BboxResize(results['img_shape'],
                                                         results['scale_factor'],
                                                         self.bbox_clip_border,
                                                         results['scale'],
                                                         self.keep_ratio))
        return results


@PIPELINES.register_module()
class RandomFlipBox(RandomFlip):
    class BboxFlip(object):
        def __init__(self, img_shape, direction):
            self.img_shape = img_shape
            self.direction = direction

        def __call__(self, bboxes, labels, masks=None):
            assert bboxes.shape[-1] % 4 == 0
            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                w = self.img_shape[1]
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif self.direction == 'vertical':
                h = self.img_shape[0]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif self.direction == 'diagonal':
                w = self.img_shape[1]
                h = self.img_shape[0]
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            else:
                raise ValueError(f"Invalid flipping direction '{self.direction}'")
            if masks is None:
                return flipped, labels
            else:
                masks = masks.flip(self.direction)
                return flipped, labels, masks

    def __call__(self, results):
        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            results['bbox_transform'].append(self.BboxFlip(results['img_shape'],
                                                           results['flip_direction']))
        return results


@PIPELINES.register_module()
class RandomErasingBox(RandomErasing):
    class BboxRandomErasing(object):
        def __init__(self, x, y, w, h):
            self.xywh = [x, y, w, h]

        def __call__(self, bboxes, labels, masks=None):
            if masks is None:
                return bboxes, labels
            x, y, w, h = self.xywh
            for i in range(len(masks.masks)):
                masks.masks[i][y:y + h, x:x + w] = 0
            return bboxes, labels, masks

    def __call__(self, results):
        if random.uniform(0, 1) >= self.p:
            return results
        img = results['img']
        y, x, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
        img[y:y + h, x:x + w] = v
        results['img'] = img
        results['bbox_transform'].append(self.BboxRandomErasing(x, y, w, h))
        return results
