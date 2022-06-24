#!/usr/bin/env bash
# note: please use xonsh, instead of bash

import os
cd ../..

cur_path = os.path.abspath(os.path.dirname(__file__))
$PYTHONPATH=cur_path

for dataset in ['C2F', 'C2B', 'K2C', 'S2C']:
  print(f'===============process {dataset}==============')
  data_root = f'./dataset/{dataset}'
  for name in ['labeled_data', 'unlabeled_data', 'test_data']:
    out_dir = f'./dataset/{dataset}/{name}.json'
    data_dir = os.path.join(data_root, name)
    if dataset in ['C2F', 'C2B']:
      python tools/datasets_uda/convert_xml_to_json.py --devkit_path @(data_dir) --out-name @(out_dir) --dataset city
    else:
      python tools/datasets_uda/convert_xml_to_json.py --devkit_path @(data_dir) --out-name @(out_dir) --dataset car

