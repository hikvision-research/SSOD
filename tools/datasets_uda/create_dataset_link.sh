import os

# please change this to your own environment
prefix = '/data'

def create_folder(file_root):
  if not os.path.exists(file_root):
    os.makedirs(file_root)

cd ../..
create_folder('dataset')
cd dataset

# 1. C2F: Cityscapes as source, foggy as target, foggy as test
print('create C2F dataset symlink: ')
create_folder('C2F')
cd C2F
create_folder('labeled_data')
cd labeled_data
ln -s @(prefix)/city/VOC2007_citytrain/* .
cd ..
create_folder('unlabeled_data')
cd unlabeled_data
ln -s @(prefix)/foggycity/VOC2007_foggytrain/* .
cd ..
create_folder('test_data')
cd test_data
ln -s @(prefix)/foggycity/VOC2007_foggyval/* .
cd ../..

# 2. C2B: Cityscapes as source, BDD100k as target, BDD100k as test
print('create C2B dataset symlink: ')
create_folder('C2B')
cd C2B
create_folder('labeled_data')
cd labeled_data
ln -s @(prefix)/city/VOC2007_citytrain/* .
cd ..
create_folder('unlabeled_data')
cd unlabeled_data
ln -s @(prefix)/BDD/VOC2007_bddtrain/* .
cd ..
create_folder('test_data')
cd test_data
ln -s @(prefix)/BDD/VOC2007_bddval/* .
cd ../..

# 3. K2C: KITTI as source, Cityscapes as target, Cityscapes as test
print('create K2C dataset symlink: ')
create_folder('K2C')
cd K2C
create_folder('labeled_data')
cd labeled_data
ln -s @(prefix)/kitti/* .
cd ..
create_folder('unlabeled_data')
cd unlabeled_data
ln -s @(prefix)/city-car/VOC2007_citytrain/* .
cd ..
create_folder('test_data')
cd test_data
ln -s @(prefix)/city-car/VOC2007_cityval/* .
cd ../..

# 4. S2C: Sim10k as source, Cityscapes as target, Cityscapes as test
print('create S2C dataset symlink: ')
create_folder('S2C')
cd S2C
create_folder('labeled_data')
cd labeled_data
ln -s @(prefix)/sim/* .
cd ..
create_folder('unlabeled_data')
cd unlabeled_data
ln -s @(prefix)/city-car/VOC2007_citytrain/* .
cd ..
create_folder('test_data')
cd test_data
ln -s @(prefix)/city-car/VOC2007_cityval/* .
cd ../..
