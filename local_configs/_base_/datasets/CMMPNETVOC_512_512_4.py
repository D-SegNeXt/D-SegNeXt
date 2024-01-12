# dataset settings
dataset_type = 'CMMPNETVOCDataset'
data_root = '../data/CMMPNETVOC_1024_4'  #修改数据集路径，为相对路径   CMMPNETVOC  VOC2012  by FRC 2023-03-03

#CMMPNETVOC3
img_norm_cfg = dict(
    # mean=[105.20, 109.26, 88.26], std=[44.23, 36.57, 38.39], to_rgb=True)
    mean=[105.20, 109.26, 88.26, 242.71], std=[44.23, 36.57, 38.39, 29.01], to_rgb=True)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)  #by FRC 2023-03-03
img_scale = (512, 512)


train_pipeline = [  #训练流程
    dict(type='LoadImageFromFile', color_type='unchanged'),   #1 加载数据
    dict(type='LoadAnnotations'),     #2  加载注释信息
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),   #3 化图像和其注释大小的数据增广的流程。    图像的最大规模   高宽比。   ?? by FRC 2023-03-03
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  #裁剪  Expected size after cropping, (h, w).
    dict(type='RandomFlip', prob=0.5),  #翻转
    dict(type='PhotoMetricDistortion'),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
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
    samples_per_gpu=8,  #单个 GPU 的 Batch size
    workers_per_gpu=4,  # 单个 GPU 分配的数据加载线程数
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline),
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
