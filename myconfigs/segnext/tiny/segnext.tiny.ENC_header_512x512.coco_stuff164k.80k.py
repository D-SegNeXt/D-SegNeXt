_base_ = [
    '../../_base_/models/mscan.py',
    '../../_base_/datasets/coco-stuff164k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k_adamw.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='../data/pretrained_models/segNeXt/mscan_t.pth')),
    decode_head=dict(
        _delete_=True,
        type='EncHead',
        in_channels=[32, 64, 160, 256],
        in_index=(0, 1, 2, 3),
        channels=128,
        num_codes=32,
        use_se_loss=True,
        add_lateral=False,
        dropout_ratio=0.1,
        num_classes=171,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=8)
evaluation = dict(interval=8000, metric=['mIoU', 'mDice', 'mFscore'])
checkpoint_config = dict(by_epoch=False, interval=8000)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
