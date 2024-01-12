#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/1 14:49
# @Author  : FlyingRocCui
# @File    : DLinkNet_Head.py
# @Description : DLinkNet的C部分
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu,inplace=True)

class DecoderBlock(BaseModule):
    def __init__(self, in_channels, n_filters,init_cfg=None):
        super(DecoderBlock, self).__init__(init_cfg)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

@HEADS.register_module()
class DLinkNetHead(BaseDecodeHead):
    def __init__(self,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.decoder4 = DecoderBlock(self.in_channels[3], self.in_channels[2])
        self.decoder3 = DecoderBlock(self.in_channels[2], self.in_channels[1])
        self.decoder2 = DecoderBlock(self.in_channels[1], self.in_channels[0])
        self.decoder1 = DecoderBlock(self.in_channels[0], self.in_channels[0])

        self.finaldeconv1 = nn.ConvTranspose2d(self.channels, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, self.num_classes, 3, padding=1)

    def forward(self, inputs):
        e1, e2, e3, e4 = inputs
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # out = self.cls_seg(out)

        # out = torch.sigmoid(out)
        #
        # out[out > 0.5] = 1
        # out[out <= 0.5] = 0

        return out


# if __name__ == '__main__':
#     x1 = torch.rand(1, 256, 128, 128)
#     x2 = torch.rand(1, 512, 64, 64)
#     x3 = torch.rand(1, 1024, 32, 32)
#     x4 = torch.rand(1, 2048, 16, 16)
#     xx = [x1, x2, x3, x4]
#     model = DLinkNetHead(channels=256,
#                          in_channels = [256, 512, 1024, 2048],
#                          in_index=[0, 1, 2, 3], num_classes= 2)
#     out = model(xx)
#     print('(((((((((')