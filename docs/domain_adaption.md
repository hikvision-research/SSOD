# DAOD

In this part, we give the tutorial about domain adaptive object detection (DAOD).

## Dataset

|  Tasks            |C2F                  |C2B                 |K2C               |S2C                    |
| ----------------- | ------------------- | ------------------ | ---------------- | --------------------- |
|  Source(Labeled)  |Cityscapes           |Cityscapes          |KITTI             |Sim10k                 |
|  Target(Unlabeled)|Foggy-Cityscapes     |BDD100k-Daytime     |Cityscapes        |Cityscapes             |

## Usage

### Training

#### 1. Use labeled data to train a baseline (aka "source only" model)

Before training，please download the pretrained backbone ([vgg](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)) to `pretrained_model/backbone`.

```shell
# |---------------------|--------|---------|
# | xonsh train_gpu2.sh | config | dataset |
# |---------------------|--------|---------|
# there are three dataset for DAOD baseline: C2F, K2C, S2C
# Note that C2B share the same "source only" model with C2F
cd examples/train/xonsh
xonsh train_gpu2.sh ./configs/baseline/baseline_uda.py C2F
```

- In our implementation, we use 2-gpus to train.

- You can also run `bash train_baseline_uda.sh` in `examples/train/bash`

- After training, we organize the pretrained baseline to `pretrained_model/baseline` as follows: 

  ```shell
  pretrained_model/
  	└── baseline/
          ├── C2F.pth
          ├── K2C.pth
          └── S2C.pth
  ```

#### 2. Use labeled data + unlabeled data to train detector

```shell
## there are four adaptation tasks: C2F, C2B, K2C, S2C
## C2F and C2B share the same "source only" model
cd examples/train/xonsh
xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_uda.py C2F
```

- In our implementation, we use 8-gpus to train.
- You can also run `bash train_uda.sh` in `examples/train/bash`

### Evaluation

```shell
# change "data_name" and "checkpoint" in scripts to support different dataset and trained model
cd examples/eval
xonsh eval_uda.sh
```

## Performance

- $\dagger$ is an ideal setting, using the label distribution from unlabeled data annotations.
- mAP: AP50

#### Normal-to-foggy weather adaptation

| C2F         | mAP  | truck | car  | rider | person | train | motor | bicycle | bus  |
| ----------- | ---- | ----- | ---- | ----- | ------ | ----- | ----- | ------- | ---- |
| source only | 30.9 | 19.2  | 47.9 | 40.8  | 34.8   | 7.8   | 24.2  | 36.0    | 36.4 |
| LabelMatch  | 52.4 | 42.0  | 62.2 | 55.4  | 45.3   | 55.1  | 43.5  | 51.5    | 64.1 |

#### Small-to-large scale dataset adaptation

|  C2B                 | mAP  | truck | car  | rider | person | train | motor | bicycle | bus  |
| -------------------- | ---- | ----- | ---- | ----- | ------ | ----- | ----- | ------- | ---- |
| source only          | 28.7 | 18.3  | 50.0 | 33.3  | 35.8   | /     | 18.4  | 27.6    | 17.0 |
| LabelMatch           | 38.8 | 39.4  | 54.6 | 37.4  | 42.9   | /     | 25.7  | 29.8    | 41.7 |
| LabelMatch$^\dagger$ | 44.5 | 39.8  | 55.4 | 44.5  | 44.8   | /     | 38.6  | 41.5    | 47.1 |

#### Cross-Camera adaptation & Synthetic-to-Real adaptation

| K2C                    | AP   | S2C                        | AP   |
| ---------------------- | ---- | -------------------------- | ---- |
| source only            | 42.2 | source only                | 36.5 |
| LabelMatch             | 51.0 | LabelMatch                 | 52.7 |
| LabelMatch$^{\dagger}$ | 52.2 | LabelMatch$^{\dagger}$     | 53.8 |

