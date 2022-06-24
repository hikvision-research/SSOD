seed = seed_template
percent = percent_template
gpu = gpu_template
samples_per_gpu = 2
total_iter = 160000
update_interval = 1000
test_interval = 2000
save_interval = 20000
# # -------------------------dataset------------------------------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

image_size = (1024, 1024)
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=image_size, ratio_range=(0.2, 1.8), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AugmentationUT', use_re=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

pipeline_u_share = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
]

pipeline_u = [
    dict(type='AddBBoxTransform'),
    dict(type='ResizeBox', img_scale=[(1333, 500), (1333, 800)], keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'bbox_transform'))
]

pipeline_u_1 = [
    dict(type='AddBBoxTransform'),
    dict(type='ResizeBox', img_scale=image_size, ratio_range=(0.2, 1.8), keep_ratio=True),
    dict(type='AugmentationUT', use_re=True, use_box=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'bbox_transform'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
data_root = './dataset/coco/'
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        type='SemiDataset',
        ann_file=data_root + f'annotations/semi_supervised/instances_train2017.{seed}@{percent}.json',
        ann_file_u=data_root + f'annotations/semi_supervised/instances_train2017.{seed}@{percent}-unlabeled.json',
        pipeline=pipeline, pipeline_u_share=pipeline_u_share,
        pipeline_u=pipeline_u, pipeline_u_1=pipeline_u_1,
        img_prefix=data_root + 'train2017/', img_prefix_u=data_root + 'train2017/',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=test_interval, metric='bbox', by_epoch=False, classwise=True)

# # -------------------------schedule------------------------------
learning_rate = 0.02 * samples_per_gpu * gpu / 32
optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[total_iter]
)
runner = dict(type='SemiIterBasedRunner', max_iters=total_iter)

checkpoint_config = dict(interval=save_interval)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
labelmatch_hook_cfg = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    label_file=data_root + f'annotations/semi_supervised/instances_train2017.{seed}@{percent}.json',
    evaluation=dict(interval=update_interval, metric='bbox', by_epoch=False),
    data=dict(
        type='TXTDataset', img_prefix=data_root + 'train2017/',
        ann_file=data_root + f'annotations/semi_supervised_txt/instances_train2017.{seed}@{percent}-unlabeled.txt',
        pipeline=test_pipeline, manual_length=10000
    )
)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='LabelMatchHook', cfg=labelmatch_hook_cfg)
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
load_from = f'./pretrained_model/baseline/instances_train2017.{seed}@{percent}.pth'
workflow = [('train', 1)]

# # -------------------------model------------------------------
model = dict(
    type='LabelMatch',
    ema_config='./configs/baseline/ema_config/baseline_standard.py',
    ema_ckpt=load_from,
    cfg=dict(
        debug=False,
    ),
    pretrained='./pretrained_model/backbone/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHeadLM',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHeadLM',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerLM',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_wrt_candidates=False,
                ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSamplerLM',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            ig_weight=0.0,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
        )
    ))
