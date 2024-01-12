_base_ = [
    '../../_base_/models/segformer.py',     #模型配置  by FRC
    '../../_base_/datasets/pascal_voc12.py',  #数据集配置  by FRC
    '../../_base_/default_runtime.py',        #运行时配置，如log 等级  间隔等  by FRC
    '../../_base_/schedules/schedule_160k_adamw.py'  #计划任务，优化器学习率等  by FRC
]

# model settings  对基础配置进行扩展    by FRC
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='../data/pretrained_models/mit_b1.pth',  #by FRC 2023-03-03
    backbone=dict(
        type='mit_b1',      #主干网的类别   by FRC
        in_chans=4,
        style='pytorch'),
    decode_head=dict(
        type='SSegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.) # head 的 LR 是 backbone 的 10 倍
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',   #预热类型
                 warmup_iters=1000,   #预热迭代次数
                 warmup_ratio=1e-5,   #
                 power=1.0, min_lr=0.0000001, by_epoch=True)  #设定为按照epoch跑

data = dict(samples_per_gpu=4)   #samples_per_gpu为batch的size，即是多少个样本为一个batch

# evaluation = dict(interval=1, metric='mIoU')   #每个epoch评估一次
evaluation = dict(interval=1, metric=['mIoU', 'mDice', 'mFscore'])   #每个epoch评估一次
