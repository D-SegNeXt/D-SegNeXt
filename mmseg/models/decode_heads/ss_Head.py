#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/29 15:59
# @Author  : FlyingRocCui
# @File    : ss_Head.py
# @Description : 这里写注释内容
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead



#IncepFormer
class SpatialSelectionModule(nn.Module):
    def __init__(self):
        super(SpatialSelectionModule, self).__init__()
        #self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.activation = nn.GELU()
        # self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))

    def forward(self, x):
        atten = self.activation(x.mean(dim=1).unsqueeze(dim=1))
        feat = torch.mul(x, atten)
        feat = x + feat
        return feat

class SpatialSelectionModule2(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialSelectionModule2, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

@HEADS.register_module()
class SpatialSelectionHead(BaseDecodeHead):
    def __init__(self, embedding_dim, **kwargs):
        super(SpatialSelectionHead, self).__init__(input_transform='multiple_select', **kwargs)

        self.ssm = SpatialSelectionModule2()

        self.linear_fuse = ConvModule(
            in_channels=sum(self.in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        inputs = [self.ssm(input) for input in inputs]

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)

        x = self.linear_fuse(inputs)
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x