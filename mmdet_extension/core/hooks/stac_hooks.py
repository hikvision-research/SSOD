# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
hooks for STAC
"""
import os.path as osp
import numpy as np

import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

import mmcv
from mmcv.runner import HOOKS, Hook, get_dist_info

from mmdet.core.evaluation import EvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


@HOOKS.register_module()
class STACHook(Hook):
    def __init__(self, cfg):
        rank, world_size = get_dist_info()
        distributed = world_size > 1
        samples_per_gpu = cfg.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.pipeline = replace_ImageToTensor(cfg.data.pipeline)
        dataset = build_dataset(cfg.data, dict(test_mode=True))
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        file = cfg.data['ann_file']
        score_thr = cfg.get('score_thr', 0.9)
        eval_cfg = cfg.get('evaluation', {})
        if distributed:
            self.eval_hook = STACDistEvalHook(file, dataloader, score_thr, **eval_cfg)
        else:
            self.eval_hook = STACEvalHook(file, dataloader, score_thr, **eval_cfg)

    def before_train_epoch(self, runner):
        self.eval_hook.before_train_epoch(runner)

    def after_train_epoch(self, runner):
        self.eval_hook.after_train_epoch(runner)

    def after_train_iter(self, runner):
        self.eval_hook.after_train_iter(runner)

    def before_train_iter(self, runner):
        self.eval_hook.before_train_epoch(runner)


class STACEvalHook(EvalHook):
    def __init__(self,
                 file,
                 dataloader,
                 score_thr,
                 **eval_kwargs
                 ):
        super().__init__(dataloader, **eval_kwargs)
        self.src_file = file
        self.score_thr = score_thr
        self.dst_root = None
        self.initial_epoch_flag = True
        self.suffix = file.split('.')[-1]
        assert self.suffix in ['json', 'txt'], 'the file type not support'

        self.dataloader = dataloader
        self.CLASSES = self.dataloader.dataset.CLASSES

    def before_train_epoch(self, runner):
        if not self.initial_epoch_flag:
            return
        if self.dst_root is None:
            self.dst_root = runner.work_dir
        interval_temp = self.interval
        self.interval = 1
        if self.by_epoch:
            self.after_train_epoch(runner)
        else:
            self.after_train_iter(runner)
        self.initial_epoch_flag = False
        self.interval = interval_temp

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.evaluation_flag(runner):
            return
        self.update_file(runner, iter_base=False)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        self.update_file(runner, iter_base=True)

    def do_evaluation(self, runner):
        from mmdet_extension.apis.test import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader)
        return results

    def _write_results_json(self, results, dst_file):
        result_files = mmcv.load(self.src_file)
        if isinstance(results[0], list):
            json_results = self.dataloader.dataset._det2json(results)
        elif isinstance(results[0], tuple):
            _, json_results = self.dataloader.dataset._segm2json(results)
        elif isinstance(results[0], np.ndarray):
            json_results = self.dataloader.dataset._proposal2json(results)
        else:
            raise TypeError('invalid type of results')
        json_results = [(lambda d: d.update({'id': i, 'area': a['bbox'][2] * a['bbox'][3]}) or d)(a)
                        for i, a in enumerate(json_results) if a['score'] > self.score_thr]
        result_files['annotations'] = json_results
        mmcv.dump(result_files, dst_file)

    def _write_results_txt(self, results, dst_file):
        with open(self.src_file, 'r', encoding='utf-8') as fr, open(dst_file, 'w', encoding='utf-8') as fw:
            for i, line in enumerate(fr):
                line_info = line.strip().split()
                h, w = int(line_info[1]), int(line_info[2])
                new_line_info = line_info[:3]
                det_pred = results[i]
                for cls, boxes in enumerate(det_pred):
                    for box in boxes:
                        if box[-1] > self.score_thr:
                            box_xyxy = [int(min(max(0, box[idx]), w if idx % 2 == 0 else h)) for idx in range(4)]
                            new_line_info.extend([str(b) for b in box_xyxy] + [str(cls + 1), '0'])
                new_line_info.insert(3, str(len(new_line_info) // 6))
                new_line = ' '.join(new_line_info)
                fw.write(new_line + '\n')

    def write_results(self, results, dst_file):
        if self.suffix == 'json':
            self._write_results_json(results, dst_file)
        else:
            self._write_results_txt(results, dst_file)

    def update_file(self, runner, iter_base=False):
        if self.initial_epoch_flag:
            dst_file = osp.join(self.dst_root, f'init.{self.suffix}')
        else:
            name = f'iter_{runner.iter}.{self.suffix}' if iter_base else f'epoch_{runner.epoch + 1}.{self.suffix}'
            dst_file = osp.join(self.dst_root, name)
        results = self.do_evaluation(runner)
        self.write_results(results, dst_file)
        if iter_base:
            self.recursive_search_dataset(runner.data_loader._dataloader.dataset, dst_file)
        else:
            self.recursive_search_dataset(runner.data_loader.dataset, dst_file)

    def recursive_search_dataset(self, dataset, dst_file):
        if hasattr(dataset, 'update_ann_file'):
            dataset.update_ann_file(dst_file)
        if hasattr(dataset, 'dataset'):
            self.recursive_search_dataset(dataset.dataset, dst_file)
        elif hasattr(dataset, 'datasets'):
            for d in dataset.datasets:
                self.recursive_search_dataset(d, dst_file)


class STACDistEvalHook(STACEvalHook):
    def __init__(self,
                 file,
                 dataloader,
                 score_thr,
                 tmpdir=None,
                 gpu_collect=False,
                 broadcast_bn_buffer=True,
                 **eval_kwargs
                 ):
        super().__init__(file, dataloader, score_thr, **eval_kwargs)
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def _broadcast_bn_buffer(self, model):
        if self.broadcast_bn_buffer:
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

    def do_evaluation(self, runner):
        if self.broadcast_bn_buffer:
            self._broadcast_bn_buffer(runner.model)
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.stac_eval_hook')
        from mmdet_extension.apis.test import multi_gpu_test
        results = multi_gpu_test(runner.model, self.dataloader,
                                 tmpdir=tmpdir, gpu_collect=self.gpu_collect)
        return results

    def update_file(self, runner, iter_base=False):
        if self.initial_epoch_flag:
            dst_file = osp.join(self.dst_root, f'init.{self.suffix}')
        else:
            name = f'iter_{runner.iter}.{self.suffix}' if iter_base else f'epoch_{runner.epoch + 1}.{self.suffix}'
            dst_file = osp.join(self.dst_root, name)
        results = self.do_evaluation(runner)
        if runner.rank == 0:
            print('\n')
            self.write_results(results, dst_file)
        dist.barrier()
        if iter_base:
            self.recursive_search_dataset(runner.data_loader._dataloader.dataset, dst_file)
        else:
            self.recursive_search_dataset(runner.data_loader.dataset, dst_file)
