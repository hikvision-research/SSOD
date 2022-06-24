import os

# please change this to fit your environment
prefix_coco = '/data'
prefix_coco_ul = '/data'
prefix_voc = '/data'

def create_folder(file_root):
  if not os.path.exists(file_root):
    os.makedirs(file_root)

cd ../..
create_folder('dataset')
cd dataset

# 1. coco
print('create coco dataset symlink: ')
create_folder('coco')
cd coco
ln -s @(prefix_coco)/coco/images/* .
if prefix_coco != prefix_coco_ul:
  ln -s @(prefix_coco_ul)/coco/* .

create_folder('annotations')
cd annotations
ln -s @(prefix_coco)/coco/annotations/* .

cd ../..
print('finish coco dataset')

# 2. voc
print('create voc dataset symlink: ')
create_folder('voc')
cd voc
ln -s @(prefix_voc)/voc/12/VOCdevkit/* .
ln -s @(prefix_voc)/voc/07/VOCdevkit/* .
cd ../..
print('finish voc dataset')
