#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/31 13:44
# @Author  : FlyingRocCui
# @File    : DeepGlobe.1024_512.Voc.120e.py.py
# @Description : segFormer对deepglobe进行训练
_base_ = [
    '../../_base_/datasets/DeepGlobeVoc_1024_512_3.py' #,  #数据集配置  by FRC
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,  #by FRC 2023-03-03
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,##变化
        num_stages=4,#不变
        num_layers=[3, 6, 40, 3], #
        num_heads=[1, 2, 5, 8], #
        patch_sizes=[7, 3, 3, 3],#卷积核
        sr_ratios=[8, 4, 2, 1], #
        out_indices=(0, 1, 2, 3), #
        mlp_ratio=4, #
        qkv_bias=True, #
        drop_rate=0.0, #
        attn_drop_rate=0.0, #
        drop_path_rate=0.1, #
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../data/pretrained_models/segformer/mit_bs5.pth')),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],##
        in_index=[0, 1, 2, 3],
        channels=768,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        # ]),
        #loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight= [0.8424, 1.1575])),# 解码头(decode_head)里的损失函数的配置项。   FocalLoss、LovaszLoss、CrossEntropyLoss、DiceLoss   use_sigmoid
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

#RunTime
log_config = dict(
    interval=120,  #每 interval 个迭代Iteration输出一次log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# optimizer torch.optim下的优化器，可选[SGD, Adam, AdamW, RMSprop, Adamax]
optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.) # head 的 LR 是 backbone 的 10 倍
                                                 }))

optimizer_config = dict()
#mmcv的lr_updater
# FixedLrUpdaterHook  StepLrUpdaterHook  ExpLrUpdaterHook PolyLrUpdaterHook InvLrUpdaterHook
# CosineAnnealingLrUpdaterHook FlatCosineAnnealingLrUpdaterHook CosineRestartLrUpdaterHook CyclicLrUpdaterHook
# OneCycleLrUpdaterHook LinearAnnealingLrUpdaterHook
lr_config = dict(policy='poly',
                 warmup='linear',   #预热类型
                 warmup_iters=1500,   #预热迭代次数
                 warmup_ratio=1e-6,   #
                 power=1, min_lr=0, by_epoch=False)  #设定为按照epoch跑  warmup_by_epoch指示warmup_iters是epoch还是inters数量  by_epoch 按照epoch还是安装iter更新学习率

# lr_config = dict(policy='Fixed',
#                  warmup='constant',   #预热类型
#                  warmup_iters=4000,   #预热迭代次数
#                  warmup_ratio=1,   #
#                  by_epoch=False)

data = dict(samples_per_gpu=16, workers_per_gpu=4)   #samples_per_gpu为batch的size，即是多少个样本为一个batch

evaluation = dict(interval=1, metric=['mIoU', 'mDice', 'mFscore'])   #每个epoch评估一次

runner = dict(type='EpochBasedRunner', max_epochs=120) #按epoch的方式进行迭代runner = dict(type='EpochBasedRunner', max_epochs=30) #按epoch的方式进行迭代

# checkpoint_config = dict(by_epoch=False, interval=4000)
checkpoint_config = dict(by_epoch=True, interval=20) #每多少次迭代（跑一个batch）保存一次模型
