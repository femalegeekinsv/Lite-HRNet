log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP')

optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    # warmup=None,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

channel_cfg = dict(
    num_output_channels=12,
    dataset_joints=12,
    dataset_channel=list(range(12)),
    inference_channel=list(range(12))
)

# model settings
model = dict(
    type='TopDown',
    pretrained='work_dirs/augmentations_litehrnet_18_coco_256x256_Envisat+IC/best.pth',
    backbone=dict(
        type='LiteHRNet',
        in_channels=3,
        frozen_stages=2,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )),
            with_head=True,
        )),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=40,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process=True,
        shift_heatmap=True,
        unbiased_decoding=False,
        modulate_kernel=11),
    loss_pose=dict(type='JointsMSELoss', use_target_weight=True))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    bbox_thr=1.0,
    use_gt_bbox=True,
    image_thr=0.0,
    bbox_file='data/ESTEC_Images/kp_sampling_img/split/5shot_Envisat_train.json',
    subfolders=None
)

val_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    bbox_thr=1.0,
    use_gt_bbox=True,
    image_thr=0.0,
    bbox_file='data/ESTEC_Images/kp_sampling_img/annotations.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ColorJitter',
        brightness=0.4, contrast= 0.4, probability=0.5
        ),
    dict(
        type='RandomErase',
        probability=0.2,
        scale= (0.01,0.01)
        ),
    dict(
        type='RandomGrayScale',
        probability=0.2),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=30,
        scale_factor=0.25
        ),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score'
        ]),
]
test_pipeline = val_pipeline
data_root = '/home/kuldeep/PhD/Code/data/Envisat'
dataset_name = 'ESTEC_Images'
seed_id = 1
num_shots = 5
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='TopDownEnvisatCocoDataset',
        ann_file=f'{data_root}/{dataset_name}/kp_sampling_img/split/seed{seed_id}/{num_shots}shot_Envisat_train.json',
        img_prefix=f'{data_root}/{dataset_name}/kp_sampling_img/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownEnvisatCocoDataset',
        ann_file=f'{data_root}/{dataset_name}/kp_sampling_img/annotations.json',
        img_prefix=f'{data_root}/{dataset_name}/kp_sampling_img/',
        data_cfg=val_data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownEnvisatCocoDataset',
        ann_file=f'{data_root}/{dataset_name}/kp_sampling_img/annotations.json',
        img_prefix=f'{data_root}/{dataset_name}/kp_sampling_img/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
)