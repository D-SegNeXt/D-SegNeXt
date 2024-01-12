_base_ = [
    '../_base_/datasets/OpenPitMineVoc1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=8,
        num_stages=7,
        strides=(1, 1, 1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2, 2, 2),
        downsamples=(True, True, True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='SimpleHead',
        in_index = 6,
        in_channels = 8,
        channels=8,
        num_classes=2,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# data
data = dict(samples_per_gpu=4)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


