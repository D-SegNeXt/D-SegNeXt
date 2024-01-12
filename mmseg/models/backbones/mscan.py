#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/20 16:41
# @Author  : FlyingRocCui
# @File    : mscan.py
# @Description : 这里写注释内容

import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
from mmseg.models.builder import BACKBONES    #by FRC

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import *
#from mmcv.runner import BaseModule, auto_fp16
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
num_classes = 1000
class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        #self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)  # DWConv(hidden_features)

        self.dwconv = DWConv(hidden_features)

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    # @auto_fp16()
    def forward(self, x):  #segFormer利用了残差网络
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

#利用卷积、BN和激活函数组合对张量进行两次升通道。bias应该设定为FALSE。SegFormer利用卷积一步到位
#输入B C H W      输出：B HW C
class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):# B C H W -> B HW C
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

#单层中利用不通尺寸的kernel获取不通尺度的特征.多分枝深度可分离条状卷积捕捉多尺度上下文信息
class AttentionModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)#聚合局部信息
        #以下为多分枝
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)#1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn * u

#单层中利用不通尺寸的kernel获取不通尺度的特征.多分枝深度可分离条状卷积捕捉多尺度上下文信息
class AttentionModuleEx1(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)#聚合局部信息
        #以下为多分枝
        self.conv1_1 = nn.Conv2d(dim, dim, (3, 3), padding=(1, 1), groups=dim, dilation=1)
        self.conv1_2 = nn.Conv2d(dim, dim, (3, 3), padding=(4, 4), groups=dim, dilation=4)
        self.conv1_3 = nn.Conv2d(dim, dim, (5, 5), padding=(8, 8), groups=dim, dilation=4)

        self.conv3 = nn.Conv2d(dim, dim, 1)#1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv1_1(attn)

        attn_1 = self.conv1_2(attn)

        attn_2 = self.conv1_3(attn)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn * u

class AttentionModuleEx2(BaseModule):
    def __init__(self, dim):
        super().__init__()
        largeK = 23
        smallK = 5
        smallpad = smallK // 2
        largepad = largeK // 2
        self.conv0 = nn.Conv2d(dim, dim, smallK, padding=smallpad, groups=dim)  # 聚合局部信息
        # 以下为多分枝
        self.conv1_1 = nn.Conv2d(dim, dim, (largeK, smallK), padding=(largepad, smallpad), groups=dim, dilation=1)
        self.conv1_2 = nn.Conv2d(dim, dim, (smallK, largeK), padding=(smallpad, largepad), groups=dim, dilation=1)
        #self.conv1_3 = nn.Conv2d(dim, dim, (5, 5), padding=(2, 2), groups=dim, dilation=1)

        self.conv3 = nn.Conv2d(dim, dim, 1)  # 1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv1_1(x)

        attn_1 = self.conv1_2(x)

        #attn_2 = self.conv1_3(attn)

        attn = attn + attn_0 + attn_1 #+ attn_2
        attn = self.conv3(attn)

        return attn * u
class AttentionModuleEx3(BaseModule):
    def __init__(self, dim):
        super().__init__()
        smallK = 5
        smallpad = smallK // 2
        self.conv0 = nn.Conv2d(dim, dim, smallK, padding=smallpad, groups=dim, dilation=1)  # 聚合局部信息
        self.conv1 = nn.Conv2d(dim, dim, smallK, padding=smallpad, groups=dim, dilation=1)  # 聚合局部信息
        self.conv2 = nn.Conv2d(dim, dim, smallK, padding=smallpad * 2, groups=dim, dilation=2)  # 聚合局部信息
        self.conv3 = nn.Conv2d(dim, dim, smallK, padding=smallpad * 4, groups=dim, dilation=4)  # 聚合局部信息
        self.conv4 = nn.Conv2d(dim, dim, 1)  # 1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_1 = self.conv1(attn)
        attn_2 = self.conv2(attn)
        attn_3 = self.conv3(attn)

        #attn_2 = self.conv1_3(attn)

        attn = attn + attn_1 + attn_2 + attn_3
        attn = self.conv4(attn)

        return attn * u

class AttentionModuleEx4(BaseModule):
    def __init__(self, dim):
        super().__init__()

        self.conv1_1 = nn.Conv2d(dim, dim, (5, 5), padding=(2, 2), groups=dim, dilation=1)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 5), padding=(4, 4), groups=dim, dilation=2)
        self.conv1_3 = nn.Conv2d(dim, dim, (5, 5), padding=(8, 8), groups=dim, dilation=4)

        self.conv2 = nn.Conv2d(dim, dim, 1)  # 1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()

        attn_1 = self.conv1_1(x)
        attn_2 = self.conv1_2(attn_1)
        attn_3 = self.conv1_3(attn_2)
        #attn = attn_1 + attn_2 + attn_3
        attn = self.conv2(attn_3)

        return attn * u
#空间注意力
class SpatialAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()

        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x

#利用具有交叠的卷积对输入进行升通道，并利用BN进行归一化，为什么没有跟激活函数呢
#输入B C H W     输出:B HW C
class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


@BACKBONES.register_module()  #by FRC
class MSCAN(BaseModule):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[32, 64, 160, 256],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.0,
                 depths=[2, 2, 2, 2],
                 updepths=[],
                 num_stages=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained_file=None,
                 init_cfg=None,
                 **kwargs):
        super(MSCAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained_file), 'init_cfg and pretrained cannot be set at the same time'

        if isinstance(pretrained_file, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, ' 'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained_file)
        elif pretrained_file is not None:
            raise TypeError('pretrained_file must be a str or None')

        self.depths = depths  # 3 3 5 2
        self.num_stages = num_stages #4
        # 线性drop out rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_chans, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # decoder
        self.updepths = updepths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(updepths))]
        cur = 0
        for i in range(len(updepths)):
            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(updepths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += updepths[i]

            setattr(self, f"decoderBlock{4 - i}", block)
            setattr(self, f"decoderNorm{4 - i}", norm)

            if i != 3:
                setattr(self, f"decoderProj{4 - i}",
                        nn.Conv2d(embed_dims[i] + embed_dims[i + 1], embed_dims[i], 1, 1, 0))

        # classification head
        #self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()


    def init_weights(self):
            print('init cfg', self.init_cfg)
            if self.init_cfg is None:
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        trunc_normal_init(m, std=.02, bias=0.)
                    elif isinstance(m, nn.LayerNorm):
                        constant_init(m, val=1.0, bias=0.)
                    elif isinstance(m, nn.Conv2d):
                        fan_out = m.kernel_size[0] * m.kernel_size[
                            1] * m.out_channels
                        fan_out //= m.groups
                        normal_init(
                            m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
            else:

                super(MSCAN, self).init_weights()

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            #if i != self.num_stages - 1:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)  #需要需改

        # x = self.head(x.mean(dim=1))   by FRC
        # return x
        decoderOuts = []
        decoderout = None
        for i in range(len(self.updepths)):
            block = getattr(self, f"decoderBlock{i + 1}")
            norm = getattr(self, f"decoderNorm{i + 1}")
            x = outs[3 - i]
            B, C, H, W = x.shape

            if i != 0:
                proj = getattr(self, f"decoderProj{i + 1}")
                decoderout = F.interpolate(decoderout, size=(H, W), mode='bilinear', align_corners=False)
                x = torch.cat([decoderout, x], dim=1)
                x = proj(x)

            x = x.flatten(2).transpose(1, 2)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            decoderout = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            decoderOuts.insert(0, decoderout)

        if(len(decoderOuts) == 0):
            return tuple(outs)
        else:
            return tuple(decoderOuts)
        #
        #return outs

class DWConv(BaseModule):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

#@register_model  by FRC
def MSCAN_Tiny(**kwargs):
    model = MSCAN(
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        # drop_rate=0.0,
        # drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     model = load_model_weights(model, "van_b0", kwargs)
    return model


#@register_model  by FRC
def MSCAN_Small(pretrained=False, **kwargs):
    model = MSCAN(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        # drop_rate=0.0,
        # drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     model = load_model_weights(model, "van_b0", kwargs)
    return model

#@register_model  by FRC
def MSCAN_Base(pretrained=False, **kwargs):
    model = MSCAN(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        # drop_rate=0.0,
        # drop_path_rate=0.1,
        depths=[3, 3, 12, 3],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     model = load_model_weights(model, "van_b0", kwargs)
    return model

#@register_model  by FRC
def MSCAN_Large(pretrained=False, **kwargs):
    model = MSCAN(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        # drop_rate=0.0,
        # drop_path_rate=0.1,
        depths=[3, 5, 27, 3],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     model = load_model_weights(model, "van_b0", kwargs)
    return model