# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
"""
convert coco json format to txt format: in order to save memory
"""
import os
import argparse
import mmcv

from mmdet_extension.core.utils.classes import COCO_CLASSES_ORG, VOC_CLASSES

COCO_MAP = {}
idx = 1
for cls in COCO_CLASSES_ORG:
    if cls == 'N/A':
        continue
    COCO_MAP[cls] = idx
    idx += 1

VOC_MAP = {}
idx = 1
for cls in VOC_CLASSES:
    if cls == 'N/A':
        continue
    VOC_MAP[cls] = idx
    idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert json format to txt format')
    parser.add_argument('--json-file', default='',
                        help='json annotations')
    parser.add_argument('--txt-file', default='',
                        help='text annotations')
    parser.add_argument('--dataset', choices=['coco', 'voc'],
                        default='coco')

    args = parser.parse_args()

    json_file = args.json_file
    txt_file = args.txt_file

    txt_root = os.path.dirname(txt_file)
    if not os.path.exists(txt_root):
        os.makedirs(txt_root)

    print(f'------------process {json_file}------------')
    json_info = mmcv.load(json_file)

    # create dict(image_id: bbox)
    image2bbox = {}
    id2image = {}
    all_images = json_info['images']
    for image in all_images:
        file_name = image['file_name']
        height, width = image['height'], image['width']
        id = image['id']
        id2image[id] = (file_name, height, width)
        image2bbox[(file_name, height, width)] = []
        if 'annotations' in json_info:
            all_annotations = json_info['annotations']
        else:
            all_annotations = []

    CLASSES = COCO_CLASSES_ORG if args.dataset == 'coco' else VOC_CLASSES
    CLASSES_MAP = COCO_MAP if args.dataset == 'coco' else VOC_MAP

    for annotation in all_annotations:
        id = annotation['image_id']
        bbox = annotation['bbox']
        ignore = annotation['iscrowd']
        cls = CLASSES_MAP[CLASSES[annotation['category_id']]]
        info = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] + [cls, ignore]
        image2bbox[id2image[id]].append(info)

    with open(txt_file, 'w', encoding='utf-8') as f:
        for key, bboxes in image2bbox.items():
            line_info = [key[0], str(key[1]), str(key[2]), str(len(bboxes))]
            for bbox in bboxes:
                line_info += [str(int(b)) for b in bbox]
            f.write(' '.join(line_info) + '\n')
