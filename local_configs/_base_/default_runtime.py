# yapf:disable
log_config = dict(
    interval=60,  #每 interval 个迭代Iteration输出一次log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
        #dict(type='NeptuneLoggerHook', init_kwargs=dict(project='flyingroccui/SegFormer',api_key="ABCDEFG"))
        # dict(type='NeptuneLoggerHook', by_epoch=False, # The Wandb logger is also supported, It requires `wandb` to be installed.
        #      init_kwargs={'entity': "OpenMMLab", # The entity used to log on Wandb
        #                   'project': "MMSeg", # Project name in WandB
        #                   'config': cfg_dict})
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
