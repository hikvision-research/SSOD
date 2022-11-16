# Data Preparation

## 1. SSOD: Semi-Supervised Object Detection

We support 5 popular settings in SSOD research as listed below：

| Labeled Data       | Unlabeled Data     | Test Data     |
| ------------------ | ------------------ | ------------- |
| COCO2017-train-1%  | COCO2017-train-99% | COCO2017-test |
| COCO2017-train-5%  | COCO2017-train-95% | COCO2017-test |
| COCO2017-train-10% | COCO2017-train-90% | COCO2017-test |
| COCO2017-train     | COCO2017-unlabeled | COCO2017-test |
| VOC07-trainval     | VOC12-trainval     | VOC07-test    |

1. Download [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](https://cocodataset.org/#home) from the website and organize them as follows:

   ```shell
   # ====coco====                      |    # ====voc====
   /data/coco/                         |    /data/voc/
     - images                          |     - 12
       - train2017                     |       - VOCdevkit
       - unlabeled2017                 |         - VOC2012
   	   - ...                           |                
     - annotations                     |     - 07
   	   - instances_train2017.json      |       - VOCdevkit
   	   - image_info_unlabeled2017.json |         - VOC2007
   	   - ...						   |						
   ```

2. Run scripts to  create the soft symlink：

   ```shell
   # * please change the "prefix_coco", "prefix_coco_ul", "prefix_voc" in the scripts to fit your environment.
   # * you can also create symlink by yourself.
   cd tools/datasets
   xonsh create_dataset_link.sh
   ```

3. Create coco-standard, coco-additional, voc (it will cost several minutes):

   ```shell
   cd tools/datasets
   xonsh preprocess_dataset.sh
   ```

## 2. DAOD: Domain Adaptive Object Detection

We support 4 popular settings in DAOD research as listed below:

|                          | Labeled Data       | Unlabeled Data           | Test Data              |
| ------------------------ | ------------------ | ------------------------ | ---------------------- |
| normal$\to$foggy (C2F)   | cityscapes (train) | cityscapes-foggy (train) | cityscapes-foggy (val) |
| small$\to$large (C2B)    | cityscapes (train) | BDD100K (train)          | BDD100K (val)          |
| across cameras (K2C)     | KITTI (train)      | cityscapes (train)       | cityscapes (val)       |
| synthetic$\to$real (S2C) | Sim10K             | cityscapes (train)       | cityscapes (val)       |

1. Download [cityscapes](https://cityscapes-dataset.com), [cityscapes-foggy](https://cityscapes-dataset.com), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d), [Sim10K](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) and [BDD100K](https://bdd-data.berkeley.edu) from the website and organize them as follows:

   ```shell
   # cityscapes          |    # cityscapes-foggy      |   # BDD
   /data/city            |    /data/foggycity         |   /data/BDD
     - VOC2007_citytrain |      - VOC2007_foggytrain  |     - VOC2007_bddtrain
       - ImageSets       |        - ImageSets         |       - ImageSets
       - JPEGImages      |        - JPEGImages        |       - JPEGImages
       - Annotations     |        - Annotations       |       - Annotations 
     - VOC2007_cityval   |      - VOC2007_foggyval    |     - VOC2007_bddval 
       - ImageSets       |        - ImageSets         |       - ImageSets
       - JPEGImages      |        - JPEGImages        |       - JPEGImages
       - Annotations     |        - Annotations       |       - Annotations 
   # =========================================================================
   # KITTI               |   # Sim10K
   /data/kitti           |   /data/sim
      - ImageSets        |     - ImageSets
      - JPEGImages       |     - JPEGImages
      - Annotations      |     - Annotations
   ```

   > PS: please refer to [ProbabilisticTeacher](https://github.com/HIK-LAB/ProbabilisticTeacher) for the detailed dataset pre-processing.

2. Run scripts to  create the soft symlink：

   ```shell
   cd tools/datasets_uda
   xonsh create_dataset_link.sh
   ```

3. Convert to coco format:

   ```bash
   cd tools/datasets_uda
   xonsh preprocess_dataset.sh
   ```

