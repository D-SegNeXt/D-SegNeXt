_base_ = [
    '../_base_/datasets/CMMPNETVOC512_3.py',  #数据集配置  by FRC
]

# model settings  SWIN  by FRC 2023-03-13
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None, #none
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
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

#RunTime
log_config = dict(
    interval=60,  #每 interval 个迭代Iteration输出一次log
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

# optimizer
optimizer = dict(type='AdamW', lr=0.002, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.) # head 的 LR 是 backbone 的 10 倍
                                                 }))
optimizer_config = dict()

# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',   #预热类型
                 warmup_iters=3000,   #预热迭代次数
                 warmup_ratio=1e-6,   #
                 power=1.0, min_lr=0, by_epoch=False)  #设定为按照epoch跑

data = dict(samples_per_gpu=16, workers_per_gpu=4)   #samples_per_gpu为batch的size，即是多少个样本为一个batch

# evaluation = dict(interval=1, metric='mIoU')   #每个epoch评估一次
evaluation = dict(interval=1, metric=['mIoU', 'mDice', 'mFscore'])   #每个epoch评估一次

# runner = dict(type='IterBasedRunner', max_iters=16000)
runner = dict(type='EpochBasedRunner', max_epochs=30) #按epoch的方式进行迭代

# checkpoint_config = dict(by_epoch=False, interval=4000)
checkpoint_config = dict(by_epoch=True, interval=1) #每多少次迭代（跑一个batch）保存一次模型