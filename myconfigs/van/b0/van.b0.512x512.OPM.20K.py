_base_ = [
    '../../_base_/models/van.py',
    '../../_base_/datasets/OpenPitMineVoc1024x1024.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k_adamw.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b2.pth'),
        drop_path_rate=0.2),
    #neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=2))


data = dict(samples_per_gpu=8)


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