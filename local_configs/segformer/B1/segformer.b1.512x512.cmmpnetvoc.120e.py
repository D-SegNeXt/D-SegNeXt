_base_ = [
    '../../_base_/models/segformer.py',     #模型配置  by FRC
    '../../_base_/datasets/CMMPNETVOC512_3.py' #,  #数据集配置  by FRC
    # '../../_base_/default_runtime.py',        #运行时配置，如log 等级  间隔等  by FRC
    # '../../_base_/schedules/schedule_320k.py'  #计划任务，优化器学习率等  by FRC
]

# model settings  对基础配置进行扩展    by FRC
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='None',#'../data/pretrained_models/mit_b0.pth',  #by FRC 2023-03-03
    backbone=dict(
        type='mit_b1',      #主干网的类别   by FRC
        in_chans=3,
        style='pytorch'),
    decode_head=dict(
        type='SSegformerHead',   # 解码头(decode head)的类别   by FRC
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,  # 解码里调整大小(resize)的 align_corners 参数。
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight= [0.8424, 1.1575])),# 解码头(decode_head)里的损失函数的配置项。   FocalLoss、LovaszLoss、CrossEntropyLoss、DiceLoss   use_sigmoid
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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

data = dict(samples_per_gpu=16, workers_per_gpu=4)   #samples_per_gpu为batch的size，即是多少个样本为一个batch


evaluation = dict(interval=1, metric=['mIoU', 'mDice', 'mFscore'])   #每个epoch评估一次

runner = dict(type='EpochBasedRunner', max_epochs=120) #按epoch的方式进行迭代runner = dict(type='EpochBasedRunner', max_epochs=30) #按epoch的方式进行迭代

# checkpoint_config = dict(by_epoch=False, interval=4000)
checkpoint_config = dict(by_epoch=True, interval=1) #每多少次迭代（跑一个batch）保存一次模型