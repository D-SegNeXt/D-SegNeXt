#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 9:46
# @Author  : FlyingRocCui
# @File    : dlinknet.py
# @Description : 这里写注释内容
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,  #by FRC 2023-03-03
    backbone=dict(
        type='DLinkNet',
        resnetLayers=101,
        init_cfg=None),
    decode_head=dict(
        type='DLinkNetHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
      # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))