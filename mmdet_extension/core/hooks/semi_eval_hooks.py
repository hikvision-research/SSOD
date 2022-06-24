# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
eval hook for semi-supervised:
1. without ema: same as normal evaluation
2. with ema: 1) only_ema=True: only do evaluation on ema_model
             2) only_ema=False: do evaluation on model and ema_model
"""
import os.path as osp
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.parallel import is_module_wrapper
from mmdet.core.evaluation import EvalHook


class SemiEvalHook(EvalHook):
    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 only_ema=False,
                 **eval_kwargs
                 ):
        super().__init__(dataloader, start, interval, by_epoch, save_best, rule,
                         **eval_kwargs)
        self.only_ema = only_ema

    def evaluation_once(self, runner):
        from mmdet_extension.apis import single_gpu_test
        if is_module_wrapper(runner.model):
            has_ema = hasattr(runner.model.module, 'ema_model') and runner.model.module.ema_model is not None
        else:
            has_ema = hasattr(runner.model, 'ema_model') and runner.model.ema_model is not None
        if (not has_ema) or (not self.only_ema):
            results = single_gpu_test(runner.model, self.dataloader, show=False)
            key_score = self.evaluate(runner, results)
        if has_ema:
            if is_module_wrapper(runner.model):
                results_ema = single_gpu_test(runner.model.module.ema_model, self.dataloader, show=False)
            else:
                results_ema = single_gpu_test(runner.model.ema_model, self.dataloader, show=False)
            key_score = self.evaluate(runner, results_ema)
        if self.save_best:
            self.save_best_checkpoint(runner, key_score)

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.evaluation_flag(runner):
            return
        self.evaluation_once(runner)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        self.evaluation_once(runner)


class SemiDistEvalHook(SemiEvalHook):
    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 tmpdir=None,
                 gpu_collect=False,
                 save_best=None,
                 rule=None,
                 only_ema=False,
                 broadcast_bn_buffer=True,
                 **eval_kwargs
                 ):
        super().__init__(dataloader, start=start, interval=interval, by_epoch=by_epoch,
                         save_best=save_best, rule=rule, only_ema=only_ema, **eval_kwargs)
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def _broadcast_bn_buffer(self, runner, has_ema):
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)
            if has_ema:
                for name, module in model.module.ema_model.named_modules():
                    if isinstance(module, _BatchNorm) and module.track_running_stats:
                        dist.broadcast(module.running_var, 0)
                        dist.broadcast(module.running_mean, 0)

    def evaluation_once(self, runner):
        has_ema = hasattr(runner.model.module, 'ema_model') and runner.model.module.ema_model is not None
        if self.broadcast_bn_buffer:
            self._broadcast_bn_buffer(runner, has_ema)
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        from mmdet_extension.apis import multi_gpu_test
        if (not has_ema) or (not self.only_ema):
            results = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect)
            if runner.rank == 0:
                print('\n')
                key_score = self.evaluate(runner, results)
        dist.barrier()
        if has_ema:
            results_ema = multi_gpu_test(
                runner.model.module.ema_model, self.dataloader,
                tmpdir=tmpdir, gpu_collect=self.gpu_collect)
            if runner.rank == 0:
                print('\n')
                key_score = self.evaluate(runner, results_ema)
        if runner.rank == 0 and self.save_best:
            self.save_best_checkpoint(runner, key_score)
