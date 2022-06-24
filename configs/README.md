## Configs

> NOTE: more detail about different **training settings** can be found in supplementary materials.

### baseline

| file                 | stage                                                   | support data_name                    |
| -------------------- | ------------------------------------------------------- | ------------------------------------ |
| baseline_ssod.py     | Training-1: use labeled data to train a baseline (SSOD) | voc, coco-standard, coco-additional  |
| baseline_uda.py      | Training-1: use labeled data to train a baseline (DAOD) | C2F, K2C, S2C              |
| baseline_uda_test.py | Evaluation (DAOD)                                       | C2F, C2B, K2C, S2C |

### labelmatch

| file                         | stage                                                        | training setting |
| ---------------------------- | ------------------------------------------------------------ | ---------------- |
| labelmatch_voc.py            | Training-2: use labeled data + unlabeled data to train detector (VOC) | VOC              |
| labelmatch_standard.py       | Training-2: use labeled data + unlabeled data to train detector (COCO-standard) | Ablation         |
| labelmatch_standard_paper.py | Training-2: use labeled data + unlabeled data to train detector (COCO-standard) | COCO-standard    |
| labelmatch_additional.py     | Training-2: use labeled data + unlabeled data to train detector (COCO-additional) | COCO-additional  |
| labelmatch_uda.py            | Training-2: use labeled data + unlabeled data to train detector (DAOD) | DAOD            |
| labelmatch_uda_prior.py      | Training-2: use labeled data + unlabeled data to train detector (DAOD, ideal setting) | DAOD            |

### others

| file                                          | stage                                                        | training setting |
| --------------------------------------------- | ------------------------------------------------------------ | ---------------- |
| stac/stac_standard.py                         | Training-2: use labeled data + unlabeled data to train detector (COCO-standard) | Ablation         |
| unbiased_teacher/unbiased_teacher_standard.py | Training-2: use labeled data + unlabeled data to train detector (COCO-standard) | Ablation         |
| soft_teacher/soft_teacher_standard.py         | Training-2: use labeled data + unlabeled data to train detector (COCO-standard) | Ablation         |