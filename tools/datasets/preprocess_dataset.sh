#!/usr/bin/env bash
# note: please use xonsh, instead of bash

import os
cd ../..

cur_path = os.path.abspath(os.path.dirname(__file__))
$PYTHONPATH=cur_path

# step1: create coco-standard (1%, 5%, 10%)
coco_data_dir = './dataset/coco'
for seed in [1, 2, 3, 4, 5]:
  for percent in [1, 5, 10]:
    python tools/datasets/prepare_coco_standard.py --data-dir @(coco_data_dir) --percent @(percent) --seed @(seed)

# step2: create coco-additional
additional_json  = './dataset/coco/annotations/image_info_unlabeled2017.json'
standard_json = './dataset/coco/annotations/instances_val2017.json'
output_json = './dataset/coco/annotations/semi_supervised/unlabeled2017.json'
python tools/datasets/convert_coco_additional.py --additional-json @(additional_json) \
--standard-json @(standard_json) --output-json @(output_json)

# step3: create voc
voc_data_dir = './dataset/voc'
out_dir = './dataset/voc/annotations_json'
python tools/datasets/convert_xml_to_json.py --devkit_path @(voc_data_dir) --out-dir @(out_dir)

# step4: (optional) convert json file to txt file
# coco
json_dir = './dataset/coco/annotations/semi_supervised'
txt_dir = './dataset/coco/annotations/semi_supervised_txt'
unlabeled_json_list = [f for f in os.listdir(json_dir) if f.find('unlabeled')!=-1]
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)
for unlabeled_json in unlabeled_json_list:
  json_file = os.path.join(json_dir, unlabeled_json)
  txt_file = os.path.join(txt_dir, unlabeled_json.replace('.json', '.txt'))
  python tools/datasets/convert_json_to_txt.py --json-file @(json_file) --txt-file @(txt_file)

# voc
txt_dir = './dataset/voc/annotations_txt'
json_file_list = os.listdir(out_dir)
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)
for json_name in json_file_list:
  json_file = os.path.join(out_dir, json_name)
  txt_file = os.path.join(txt_dir, json_name.replace('.json', '.txt'))
  python tools/datasets/convert_json_to_txt.py --json-file @(json_file) --txt-file @(txt_file) --dataset voc