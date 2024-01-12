#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/1 13:43
# @Author  : FlyingRocCui
# @File    : DLinkNet.py
# @Description : DLinkNet的AB两部分
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(curPath)[0])

import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

# from ..builder import BACKBONES
# from ..utils import ResLayer
from torchvision import models
import torch.nn.functional as F
from functools import partial
from mmseg.models.builder import BACKBONES

nonlinearity = partial(F.relu,inplace=True)

class DBlock(nn.Module):
    def __init__(self,
                 channel,
                 dilations=[1, 2, 4, 8, 16],
                 init_cfg=None):
        super(DBlock, self).__init__(init_cfg)
        self.dilations = dilations

        for i in range(len(dilations)):
            dilate = nn.Conv2d(channel, channel, kernel_size=3, dilation=dilations[i], padding=dilations[i])
            setattr(self, f"dilate{i + 1}", dilate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = x
        dilatex = x
        for i in range(len(self.dilations)):
            dilate = getattr(self, f"dilate{i + 1}")
            dilatex = nonlinearity(dilate(dilatex))
            out += dilatex

        return out

@BACKBONES.register_module()
class DLinkNet(BaseModule):
    def __init__(self,
                 resnetLayers=101,
                 init_cfg=None,
                 **kwargs):
        super(DLinkNet, self).__init__(init_cfg)

    #Part A
        resnet = None
        if resnetLayers == 34:
            resnet = models.resnet34(pretrained=True)
            self.dblock = DBlock(512)
        elif resnetLayers == 50:
            resnet = models.resnet50(pretrained=True)
            self.dblock = DBlock(2048)
        elif resnetLayers == 101:
            resnet = models.resnet101(pretrained=True)
            self.dblock = DBlock(2048)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
    #Part B

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)
        return [e1, e2, e3, e4]


# if __name__ == '__main__':
#     x = torch.rand(1, 3, 512, 512)
#     model = DLinkNet(34)
#     out34 = model(x)
#     model = DLinkNet(50)
#     out50 = model(x)
#     model = DLinkNet(101)
#     out101 = model(x)
#     print('aaaaaaaaa')