_base_ = [
    '../../_base_/models/mscan.py',
    '../../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    backbone=dict(#dcan_ex10_t
        init_cfg=dict(type='Pretrained', checkpoint='../data/pretrained_models/segNeXt/mscan_t.pth')),#'../data/pretrained_models/ImageNet17GsegNeXt/mscan.pth'
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(  # 二分类问题用sigmoid，多分类用softmax因此这里设定为true
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    #test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# data
data = dict(samples_per_gpu=4)
evaluation = dict(interval=8000, metric=['mIoU', 'mDice', 'mFscore'])
checkpoint_config = dict(by_epoch=False, interval=8000)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

#fp16 settings
#optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=5000.)
# fp16 placeholder
#fp16 = dict(loss_scale='dynamic')

#fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# optimizer_config = dict(_delete_=True,
#                         type='Fp16OptimizerHook', loss_scale='dynamic',
#                         grad_clip=dict(max_norm=35, norm_type=2))
#fp16 placeholder
#fp16 = dict()  #loss_scale='dynamic'


#fp16 = dict(loss_scale=512.) # 表示静态 scale
# 表示动态 scale
#fp16 = dict(loss_scale='dynamic')

# 通过字典形式灵活开启动态 scale
#fp16 = dict(loss_scale=dict(init_scale=512.,mode='dynamic'))