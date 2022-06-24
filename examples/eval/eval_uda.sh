#!/usr/bin/env bash
# note: please use xonsh, instead of bash

import os
cd ../..
$LANG='zh_CN.UTF-8'
$LANGUAGE='zh_CN:zh:en_US:en'
$LC_ALL='C.UTF-8'

cur_path = os.path.abspath(os.path.dirname(__file__))
$PYTHONPATH=cur_path

GPU = 2
data_name='city'
checkpoint=f'./pretrained_model/baseline/city.pth'

config='./configs/baseline/baseline_uda_test.py'
eval_type='bbox'

new_config = config[:-3] + f'_{data_name}.py'
python tools/create_config_from_template.py --org-config @(config) --new-config @(new_config) \
--data @(data_name) --gpu @(GPU)

if GPU>1:
  python -m torch.distributed.launch --nproc_per_node=@(GPU) --master_port=19005 \
  examples/eval/eval.py --config @(new_config) --checkpoint @(checkpoint) --launcher pytorch \
  --eval @(eval_type)
else:
  python examples/eval/eval.py --config @(new_config) --checkpoint @(checkpoint) --eval @(eval_type)

os.remove(new_config)