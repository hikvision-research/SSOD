# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.samplers import RandomSampler, SamplingResult
from mmdet_extension.core.bbox.samplers.sampling_result_lm import SamplingResultLM


@BBOX_SAMPLERS.register_module()
class RandomSamplerLM(RandomSampler):
    def _sample_neg_ig(self, assign_result, assign_result_ig, num_expected, **kwargs):
        pos_inds, ig_inds = assign_result.gt_inds, assign_result_ig.gt_inds
        if len(pos_inds) != len(ig_inds):
            ig_inds = torch.cat([pos_inds.new_ones(len(pos_inds) - len(ig_inds)), ig_inds])
        neg_inds = torch.nonzero((pos_inds == 0) & (ig_inds == 0), as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    # NOTE: here the start index is not same as pos and neg
    def _sample_ignore(self, assign_result, assign_result_ig, num_expected, **kwargs):
        pos_inds, ig_inds = assign_result.gt_inds, assign_result_ig.gt_inds
        if len(pos_inds) != len(ig_inds):
            pos_inds = pos_inds[-len(ig_inds):]
        select_inds = torch.nonzero((pos_inds <= 0) & (ig_inds > 0), as_tuple=False)
        if select_inds.numel() != 0:
            select_inds = select_inds.squeeze(1)
        if len(select_inds) <= num_expected:
            return select_inds
        else:
            return self.random_choice(select_inds, num_expected)

    def sample_ig(self,
                  assign_result,
                  assign_result_ig,
                  bboxes,
                  gt_bboxes,
                  gt_labels=None,
                  **kwargs):
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]
        bboxes = bboxes[:, :4]
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            assign_result_ig.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_ig = num_expected_pos - num_sampled_pos
        ig_inds = self.pos_sampler._sample_ignore(
            assign_result, assign_result_ig, num_expected_ig, bboxes=bboxes, **kwargs
        )
        ig_inds = ig_inds.unique()
        num_sampled_ig = ig_inds.numel()
        num_expected_neg = self.num - num_sampled_pos - num_sampled_ig
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg_ig(
            assign_result, assign_result_ig, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        if kwargs.get('with_ignore', False):
            return sampling_result, ig_inds
        else:
            return sampling_result

    def sample_pos_ig(self,
                      assign_result,
                      assign_result_ig,
                      bboxes,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_labels_ignore=None,
                      **kwargs):
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]
        bboxes = bboxes[:, :4]
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError('gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            assign_result_ig.add_ig_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        if self.add_gt_as_proposals and len(gt_bboxes_ignore) > 0:
            bboxes = torch.cat([gt_bboxes_ignore, bboxes], dim=0)
            assign_result.add_ig_(gt_labels_ignore)
            assign_result_ig.add_gt_(gt_labels_ignore)
            gt_ones = bboxes.new_ones(gt_bboxes_ignore.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        num_expected_pos = int(self.num * self.pos_fraction)
        # sample pos
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        # sample ignore
        num_expected_ig = num_expected_pos - num_sampled_pos
        ig_inds = self.pos_sampler._sample_pos(
            assign_result_ig, num_expected_ig, bboxes=bboxes, **kwargs)
        ig_inds = ig_inds.unique()
        num_sampled_ig = ig_inds.numel()
        # sample negative
        num_expected_neg = self.num - num_sampled_pos - num_sampled_ig
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg_ig(
            assign_result, assign_result_ig, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()
        sampling_result = SamplingResultLM(
            pos_inds, ig_inds, neg_inds, bboxes, gt_bboxes,
            gt_bboxes_ignore, assign_result, assign_result_ig, gt_flags)
        return sampling_result
