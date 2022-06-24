# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Support save ema model
"""
import os.path as osp
import platform
import shutil

import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint

from mmcv.parallel import is_module_wrapper
from mmcv.runner import IterBasedRunner, EpochBasedRunner


@RUNNERS.register_module()
class SemiIterBasedRunner(IterBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        self.call_hook('before_train_iter')
        data_batch = next(data_loader)
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='iter_{}.pth',
                        meta=None,
                        save_optimizer=True,
                        create_symlink=True):
        if meta is None:
            meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
        elif isinstance(meta, dict):
            meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        filepath_ema = filepath[:-4] + '_ema.pth'
        if is_module_wrapper(self.model):
            use_ema = hasattr(self.model.module, 'ema_model') and self.model.module.ema_model is not None
            if use_ema:
                save_checkpoint(self.model.module.ema_model, filepath_ema, optimizer=optimizer, meta=meta)
        else:
            use_ema = hasattr(self.model, 'ema_model') and self.model.ema_model is not None
            if use_ema:
                save_checkpoint(self.model.ema_model, filepath_ema, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class SemiEpochBasedRunner(EpochBasedRunner):
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='iter_{}.pth',
                        meta=None,
                        save_optimizer=True,
                        create_symlink=True):
        if meta is None:
            meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
        elif isinstance(meta, dict):
            meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        filepath_ema = filepath[:-4] + '_ema.pth'
        if is_module_wrapper(self.model):
            use_ema = hasattr(self.model.module, 'ema_model') and self.model.module.ema_model is not None
            if use_ema:
                save_checkpoint(self.model.module.ema_model, filepath_ema, optimizer=optimizer, meta=meta)
        else:
            use_ema = hasattr(self.model, 'ema_model') and self.model.ema_model is not None
            if use_ema:
                save_checkpoint(self.model.ema_model, filepath_ema, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
