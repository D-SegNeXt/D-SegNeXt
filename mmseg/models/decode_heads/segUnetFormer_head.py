#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 9:32
# @Author  : FlyingRocCui
# @File    : segUnetFormer_head.py
# @Description : 这里写注释内容
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr
from einops import rearrange
from IPython import embed

from mmseg.models.backbones.segUnetFormer import *

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

# 总体膨胀scale倍数，分辨率也膨胀scale倍，通道数减少scale倍数
class PatchExpand(nn.Module):
    def __init__(self, scale, hwScale, channel, norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = scale
        self.channel = channel
        self.hwScale = hwScale
        self.linear = nn.Linear(channel, channel * scale)
        self.norm = norm_layer(channel * scale // (hwScale ** 2))

    def forward(self, x):
        B, C, H, W = x.shape
        xnew = x.permute(0,2,3,1).view(B, -1, C)
        xnew = self.linear(xnew)
        xnew = xnew.view(B, H, W, C * self.scale)
        xnew = rearrange(xnew, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.hwScale, p2=self.hwScale, c=C * self.scale //(self.hwScale ** 2))
        x = self.norm(xnew.clone())
        x = x.permute(0,3,1,2).contiguous()
        return x

#合并两个分辨率相同的张量，
class catTensor(nn.Module):
    def __init__(self, inchannel, outchannel, norm_layer=nn.LayerNorm):
        super().__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.linear = nn.Linear(inchannel, outchannel)

    def forward(self, x1, x2):
        b1, c1, h1, w1 = x1.shape
        b2, c2, h2, w2 = x2.shape
        assert h1 == h2 & w1 == w2
        x1 = x1.view(b1, -1, c1)
        x2 = x2.view(b2, -1, c2)
        catx = torch.cat([x1, x2], dim=-1)
        catx = self.linear(catx).contiguous()
        return catx, h1, w1


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act_layer=nn.GELU):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * out_ch, out_ch, 3, padding=1),
            act_layer()
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

@HEADS.register_module()
class SegUnetFormerHeadL0(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    # class SegUnetFormerHeadB0(SegUnetFormerHead):
    #     def __init__(self, **kwargs):
    #         super(SegUnetFormerHeadB0, self).__init__(
    #             embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    #             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
    #             drop_rate=0.0, drop_path_rate=0.1, in_channels=[32, 64, 160, 256], channels=128, num_classes=2,
    #             **kwargs)

    # def __init__(self, feature_strides, **kwargs):
    #     super().__init__(input_transform='multiple_select', **kwargs)
    #     assert len(feature_strides) == len(self.in_channels)
    #     assert min(feature_strides) == feature_strides[0]
    #     self.feature_strides = feature_strides

    def __init__(self, feature_strides, patch_size=4, qk_scale= None, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_drop_rate=0., drop_rate=0.0, drop_path_rate=0.1, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随着深度进行增加
        cur = 0
        # self.patch_expand1 = PatchExpand(16, 4, self.in_channels[0])
        self.cattensor1 = catTensor(self.in_channels[0] + self.in_channels[1] // 2 , self.in_channels[0])
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=self.in_channels[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(self.in_channels[0])

        cur += depths[0]
        self.patch_expand2 = PatchExpand(2, 2, self.in_channels[1])
        self.cattensor2 = catTensor(self.in_channels[1] + self.in_channels[2] // 2 , self.in_channels[1])
        self.block2 = nn.ModuleList([TransformerBlock(
            dim=self.in_channels[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(self.in_channels[1])

        cur += depths[1]
        self.patch_expand3 = PatchExpand(2, 2, self.in_channels[2])
        self.cattensor3 = catTensor(self.in_channels[2] + self.in_channels[3] // 2 , self.in_channels[2])
        self.block3 = nn.ModuleList([TransformerBlock(
            dim=self.in_channels[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(self.in_channels[2])

        cur += depths[2]
        self.patch_expand4 =  PatchExpand(2, 2, self.in_channels[3])
        self.block4 = nn.ModuleList([TransformerBlock(
            dim=self.in_channels[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(self.in_channels[3])


        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']  #256
        # embedding_dim = 256
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(self.in_channels[0], self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## Transformer decoder on C1-C4 ###########
        _c4 = self.patch_expand4(c4)

        #block 3
        _c34, H, W  = self.cattensor3(c3, _c4)
        B, _, C = _c34.shape
        for i, blk in enumerate(self.block3):
            _c34 = blk(_c34, H, W)
        _c3 = self.norm3(_c34)
        _c3 = _c3.view(B, H, W, C).permute(0, 3, 1, 2)
        _c3 = self.patch_expand3(_c3)

        # block 2
        _c23, H, W   = self.cattensor2(c2, _c3)
        B, _, C= _c23.shape
        for i, blk in enumerate(self.block2):
            _c23 = blk(_c23, H, W)
        _c2 = self.norm2(_c23)
        _c2 = _c2.view(B, H, W, C).permute(0, 3, 1, 2)
        _c2 = self.patch_expand2(_c2)

        _c12, H, W  = self.cattensor1(c1, _c2)
        B, _, C = _c12.shape
        for i, blk in enumerate(self.block1):
            _c12 = blk(_c12, H, W)
        _c1 = self.norm1(_c12)
        _c1 = _c1.view(B, H, W, C).permute(0, 3, 1, 2)

        # _c1 = self.patch_expand1(_c1)
        x = self.dropout(_c1)
        x = self.linear_pred(x)

        return x

@HEADS.register_module()
class SegUnetFormerHead(BaseDecodeHead):

    def __init__(self, feature_strides, patch_size=4, qk_scale= None, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_drop_rate=0., drop_rate=0.0, drop_path_rate=0.1, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.upblock1 = UpBlock(self.in_channels[3], self.in_channels[2])
        self.upblock2 = UpBlock(self.in_channels[2], self.in_channels[1])
        self.upblock3 = UpBlock(self.in_channels[1], self.in_channels[0])
        self.upblock4 = UpBlock(self.in_channels[0], self.num_classes)

        self.conv = nn.Conv2d(self.in_channels[0], self.num_classes, 1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        x = self.upblock1(c4, c3)
        x = self.upblock2(x, c2)
        x = self.upblock3(x, c1)
        # x = self.upblock4(x)
        x = self.conv(x)

        return x



@HEADS.register_module()
class SegUnetFormerHeadB0(SegUnetFormerHead):
    def __init__(self, **kwargs):
        super(SegUnetFormerHeadB0, self).__init__(
            patch_size=4, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_drop_rate=0., drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@HEADS.register_module()
class SegUnetFormerHeadB1(SegUnetFormerHead):
    def __init__(self, **kwargs):
        super(SegUnetFormerHeadB1, self).__init__(
            patch_size=4, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_drop_rate=0., drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@HEADS.register_module()
class SegUnetFormerHeadB2(SegUnetFormerHead):
    def __init__(self, **kwargs):
        super(SegUnetFormerHeadB2, self).__init__(
            patch_size=4, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@HEADS.register_module()
class SegUnetFormerHeadB3(SegUnetFormerHead):
    def __init__(self, **kwargs):
        super(SegUnetFormerHeadB3, self).__init__(
            patch_size=4, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            attn_drop_rate=0., drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@HEADS.register_module()
class SegUnetFormerHeadB4(SegUnetFormerHead):
    def __init__(self, **kwargs):
        super(SegUnetFormerHeadB4, self).__init__(
            patch_size=4, inum_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            attn_drop_rate=0., drop_rate=0.0, drop_path_rate=0.1, **kwargs)

@HEADS.register_module()
class SegUnetFormerHeadB5(SegUnetFormerHead):
    def __init__(self, **kwargs):
        super(SegUnetFormerHeadB5, self).__init__(
            patch_size=4, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            attn_drop_rate=0., drop_rate=0.0, drop_path_rate=0.1, **kwargs)


