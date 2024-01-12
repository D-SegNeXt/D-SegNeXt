_base_ = [
    '../_base_/datasets/CMMPNETVOC512_4.py',  #数据集配置  by FRC
]

# model settings  ASPPUNET  by FRC 2023-03-13
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='../data/pretrained_models/unet/fcn_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf_20211210_201821-c314da8a.pth', #none
    backbone=dict(
        type='UNet',
        in_channels=4,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='ASPPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
    # test_cfg=dict(mode='slide', crop_size=256, stride=170))

# optimizer
# optimizer = dict(_delete_=True, type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01,
#                  paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.),
#                                                  'head': dict(lr_mult=10.) # head 的 LR 是 backbone 的 10 倍
#                                                  }))

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
optimizer = dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01,
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
                 warmup_iters=3000,   #预热迭代次数
                 warmup_ratio=1e-6,   #
                 power=1, min_lr=0, by_epoch=False)  #设定为按照epoch跑  warmup_by_epoch指示warmup_iters是epoch还是inters数量  by_epoch 按照epoch还是安装iter更新学习率

# lr_config = dict(policy='Fixed',
#                  warmup='constant',   #预热类型
#                  warmup_iters=4000,   #预热迭代次数
#                  warmup_ratio=1,   #
#                  by_epoch=False)

data = dict(samples_per_gpu=8, workers_per_gpu=4)   #samples_per_gpu为batch的size，即是多少个样本为一个batch


evaluation = dict(interval=1, metric=['mIoU', 'mDice', 'mFscore'])   #每个epoch评估一次

runner = dict(type='EpochBasedRunner', max_epochs=120) #按epoch的方式进行迭代runner = dict(type='EpochBasedRunner', max_epochs=30) #按epoch的方式进行迭代

# checkpoint_config = dict(by_epoch=False, interval=4000)
checkpoint_config = dict(by_epoch=True, interval=1) #每多少次迭代（跑一个batch）保存一次模型