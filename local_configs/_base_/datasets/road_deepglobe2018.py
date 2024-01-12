# dataset settings
dataset_type = 'RoadDataset'
data_root = '../data/deepglobe-road-dataset'  #修改数据集路径，为相对路径
img_norm_cfg = dict(
    mean=[104.48, 97.62, 73.47], std=[32.88, 26.95, 24.81], to_rgb=True)

img_scale = (1024, 1024)
crop_size = (512, 512)  #by FRC 2023-03-03

train_pipeline = [  #训练流程
    dict(type='LoadImageFromFile'),   #1 加载数据
    dict(type='LoadAnnotations'),     #2  加载注释信息
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),   #3 化图像和其注释大小的数据增广的流程。    图像的原始尺寸 by FRC 2023-03-03
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  #裁剪  图像的裁剪尺寸
    dict(type='RandomFlip', prob=0.5),  #翻转
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,  # by FRC 2023-03-03
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,  #单个 GPU 的 Batch size
    workers_per_gpu=4,  # 单个 GPU 分配的数据加载线程数
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/img',
        ann_dir='train/label',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='valid/img',
        ann_dir='valid/label',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/img',
        ann_dir='test/label',
        pipeline=test_pipeline))
