#!/usr/bin/env bash
# note: please use xonsh, instead of bash

import os
cd ../../..
$LANG='zh_CN.UTF-8'
$LANGUAGE='zh_CN:zh:en_US:en'
$LC_ALL='C.UTF-8'

cur_path = os.path.abspath(os.path.dirname(__file__))
par_path = os.path.join(cur_path, '../')
$PYTHONPATH=cur_path


# #------------------------------------template for 2GPU------------------------------------
GPU = 8
config = $ARG1
second_arg = $ARG2
try:
  int(second_arg)
  seed = $ARG2
  percent = $ARG3
  name = $ARG4
  times = 1
  if name == 'coco-standard':
    times = 5 if percent == '1' else 2
  new_config = config[:-3] + f'_{seed}_{percent}_{GPU}.py'
  python tools/create_config_from_template.py --org-config @(config) --new-config @(new_config) \
  --seed @(seed) --percent @(percent) --gpu @(GPU) --data @(name) --times @(times)
except:
  name = $ARG2
  new_config = config[:-3] + f'_{name}.py'
  print(name)
  python tools/create_config_from_template.py --org-config @(config) --new-config @(new_config) \
  --data @(name) --gpu @(GPU)

if GPU > 1:
  python -m torch.distributed.launch --nproc_per_node=@(GPU) --master_port=19005 \
  examples/train/train.py --config @(new_config) --launcher pytorch
else:
  python examples/train/train.py --config @(new_config)
os.remove(new_config)