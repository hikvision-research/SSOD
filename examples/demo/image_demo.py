from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector
import numpy as np
from mmdet_extension.core.utils.image import imshow_det_bboxes
import mmdet_extension


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--img', default='./dataset/coco/val2017/000000186938.jpg',
        help='Image file')
    parser.add_argument(
        '--config', default='./configs/baseline/ema_config/baseline_standard.py', help='Config file')
    parser.add_argument(
        '--checkpoint', default='./pretrained_model/baseline/instances_train2017.1@1.pth',
        help='Checkpoint file')
    parser.add_argument(
        '--output', default=None,
        help='output image for this demo')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.6, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # test a single image
    result = inference_detector(model, args.img)

    # visualize or save the results
    bboxes, labels = [], []
    for c, r in enumerate(result):
        if len(r) > 0:
            bboxes.append(r)
            labels.append(np.array([c] * len(r)))
    bboxes = np.concatenate(bboxes)
    labels = np.concatenate(labels)
    imshow_det_bboxes(args.img, bboxes, labels.astype(np.int), class_names=model.CLASSES,
                      score_thr=args.score_thr, thickness=2, font_size=13,
                      out_file=args.output
                      )


if __name__ == '__main__':
    main()
