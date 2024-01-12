_base_ = [
    '../_base_/datasets/OpenPitMineVoc1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='NL34_LinkNet'),
    decode_head=dict(
        type='SimpleHead',
        in_index=0,
        in_channels=32,
        channels=32,
        num_classes=2,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='dice_bce_loss', loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# data
data = dict(samples_per_gpu=4)

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


