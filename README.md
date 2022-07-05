# MMDetection-based Toolbox for Semi-Supervised Object Detection

## Supported algorithms

- [x] STAC：[A Simple Semi-Supervised Learning Framework for Object Detection [Arxiv'20]](https://arxiv.org/abs/2005.04757v2)
- [x] Unbiased Teacher：[Unbiased Teacher for Semi-Supervised Object Detection [ICLR'21]](https://arxiv.org/abs/2102.09480)
- [x] Soft Teacher：[End-to-End Semi-Supervised Object Detection with Soft Teacher [ICCV'21]](https://arxiv.org/abs/2106.09018)
- [x] LabelMatch：[Label Matching Semi-Supervised Object Detection [CVPR'22]](https://arxiv.org/pdf/2206.06608.pdf)

## Preparation

#### Prerequisites

```bash
pip install -r requirements.txt
```

- Linux with Python >= 3.6
- We use mmdet=2.10.0, pytorch=1.6.0

#### Data Preparation

Please refer to [prepare_data.md](./docs/prepare_data.md).

## Usage

### Training

#### 1. Use labeled data to train a baseline

Before training，please download the pretrained backbone ([resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)) to `pretrained_model/backbone`.

```shell
# |---------------------|--------|------|---------|---------|
# | xonsh train_gpu2.sh | config | seed | percent | dataset |
# |---------------------|--------|------|---------|---------|
cd examples/train/xonsh
## ---dataset: coco-standard---
xonsh train_gpu2.sh ./configs/baseline/baseline_ssod.py 1 1 coco-standard
## ---dataset: voc---
# xonsh train_gpu2.sh ./configs/baseline/baseline_ssod.py 1 1 voc
## ---dataset: coco-additional---
# xonsh train_gpu8.sh ./configs/baseline/baseline_ssod.py 1 1 coco-additional
```

- In our implementation, we use 2-gpus to train except coco-additional.

- After training, we organize the pretrained baseline to `pretrained_model/baseline` as follows：

  ```shell
  pretrained_model/
  	└── baseline/
          ├── instances_train2017.1@1.pth
          ├── instances_train2017.1@5.pth
          ├── ...
          ├── voc.pth
          └── coco.pth
  ```

  - You can also change the `load_from` information in `config` file in step 2.

#### 2. Use labeled data + unlabeled data to train detector

```shell
## note: dataset is set to none in this step.
cd examples/train/xonsh
xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_standard.py 1 1 none
```

- In our implementation, we use 8-gpus to train.
- You can also run `bash train_ssod.sh` in `examples/train/bash`

### Evaluation

```shell
# please change "config" and "checkpoint" in 'eval.sh' scripts to support different dataset and trained model
cd examples/eval
xonsh eval.sh
```

## Performance

#### LabelMatch

| Model   | Supervision | AP   | Config | Model Weights |
| :-------: | :-----------: | :--: | :-----------: | ------------- |
| R50-FPN | 1%          | 25.81±0.28 | [labelmatch_standard_paper](./configs/labelmatch/labelmatch_standard_paper.py) | [To-Be-Released]() |
| R50-FPN | 5% | 32.70±0.18 | [labelmatch_standard_paper](./configs/labelmatch/labelmatch_standard_paper.py) | [To-Be-Released]() |
| R50-FPN | 10% | 35.49±0.17 | [labelmatch_standard_paper](./configs/labelmatch/labelmatch_standard_paper.py) | [To-Be-Released]() |

- Please refer to [performance.md](./docs/performance.md) for more performance presentation.

## Extension to Domain adaptive object detection

Please refer to [UDA](./docs/domain_adaption.md)

## Citation

If you use LabelMatch in your research or wish to refer to the results published in the paper, please consider citing out paper.

```BibTeX
@inproceedings{Chen2022LabelMatching,
    title={Label Matching Semi-Supervised Object Detection},
    author={Binbin Chen, Weijie Chen, Shicai Yang, Yunyi Xuan, JieSong, Di Xie, Shiliang Pu, Mingli Song, Yueting Zhuang.},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2022},
}
```

## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.

## Acknowledgement

If you have any problem about this work, please feel free to contact Binbin Chen (chenbinbin8-at-hikvision.com) and Weijie Chen (chenweijie5-at-hikvision.com).

