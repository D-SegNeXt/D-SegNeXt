# dataset settings
dataset_type = 'RoadDataset'
data_root = '../data/DBStepLines'  #修改数据集路径，为相对路径   CMMPNETVOC  VOC2012  by FRC 2023-03-03

#CMMPNETVOC3
img_norm_cfg = dict(
    mean=[176.25, 180.49, 97.38], std=[43.67, 37.23, 37.52], to_rgb=True)
    # mean=[154.06, 120.07, 68.15], std=[19.962, 17.83, 12.88], to_rgb=True)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)  #by FRC 2023-03-03
img_scale = (1280, 1280)


train_pipeline = [  #训练流程
    dict(type='LoadImageFromFile'),  # 1 加载数据
    dict(type='LoadAnnotations'),  # 2  加载注释信息
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    # 3 化图像和其注释大小的数据增广的流程。    图像的最大规模   高宽比。   ?? by FRC 2023-03-03
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # 代表最多类别的像素值占整个图片的比例大于- cat_max_ratio，就进行裁剪，防止背景占比太大
    dict(type='RandomFlip', prob=0.5),  # 翻转
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
            dict(type='ResizeToMultiple', size_divisor=32),
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
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='JPEGImages',
            ann_dir='SegmentationClass',
            split='ImageSets/Segmentation/train.txt',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/test.txt',
        pipeline=test_pipeline))
