#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/15 11:33
# @Author  : FlyingRocCui
# @File    : test.py
# @Description : 这里写注释内容
import torch, gc
from mmseg.models.decode_heads.segUnetFormer_head import *
from mmseg.models.backbones.unet import  *
from mmseg.models.backbones.segUnetFormer import  *
import numpy as np
import argparse
import os

import os
import json

# import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import torch
import pickle

class HDMSCA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1_1 = nn.Conv2d(dim, dim, (5, 5), padding=(2, 2), groups=dim, dilation=1)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 5), padding=(4, 4), groups=dim, dilation=2)
        self.conv1_3 = nn.Conv2d(dim, dim, (5, 5), padding=(10, 10), groups=dim, dilation=5)
        self.conv1_4 = nn.Conv2d(dim, dim, (5, 5), padding=(18, 18), groups=dim, dilation=9)

        self.conv2 = nn.Conv2d(dim, dim, 1)  # 1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()

        attn_1 = self.conv1_1(x)
        attn_2 = self.conv1_2(attn_1)
        attn_3 = self.conv1_3(attn_2)
        attn_4 = self.conv1_4(attn_3)

        attn = attn_1 + attn_2 + attn_3 + attn_4
        attn = self.conv2(attn)

        return attn * u

class MSCA(nn.Module):
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

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)

if __name__ == '__main__':
    #segNeXtTinyAttention2 = torch.load("E:/latest.pth")
    # model = SpatialAttention()
    # inp = torch.rand(1, 3, 512, 512)
    # out = model(inp)

    # pems04_data = np.load('G:/Data/PEMS/PEMS04/PEMS04.npz')
    # pems08_data = np.load('G:/Data/PEMS/PEMS08/PEMS08.npz')
    # train_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/NYCTaxi/train.npz')
    # test_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/NYCTaxi/test.npz')
    # val_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/NYCTaxi/val.npz')
    #
    # BJtrain_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/BJTaxi/train.npz')
    # BJtest_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/BJTaxi/test.npz')
    # BJval_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/BJTaxi/val.npz')
    #
    # NYCBtrain_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/NYCBike1/train.npz')
    # NYCBtest_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/NYCBike1/test.npz')
    # NYCBval_data = np.load('E:/WorkRoom/Sources/doctor/Trafic Prediction/ST-SSL_Dataset/NYCBike1/val.npz')


    #print(pems04_data.files)
    #各个模型的参数量计算
    # _List = [1, 2, 3, 4, 5, 6]
    # a = _List[-1]
    _mHDMSCA = HDMSCA(256)
    # _mMSCA = MSCA(256)
    # _mLKA = LKA(256)
    _mTotalHDMSCA = sum([param.nelement() for param in _mHDMSCA.parameters()])
    # _mTotalMSCA = sum([param.nelement() for param in _mMSCA.parameters()])
    # _mTotalLKA = sum([param.nelement() for param in _mLKA.parameters()])
    print(_mTotalHDMSCA)
    # print(_mTotalMSCA)
    # print(_mTotalLKA)

    # segNeXtTinyAttention = torch.load("E:\\Folders\\Downloads\\segnext_tiny_512x512_ade_160k.pth")
    # segNeXtTinyAttention2 = torch.load("E:\\WorkRoom\\Sources\\doctor\\logs\\20230823_143229\\iter_50.pth")
    #
    #
    # segNeXtSmallAttention = torch.load("E:\\Folders\\Downloads\\segnext_small_512x512_ade_160k.pth")
    # segNeXtBaseAttention = torch.load("E:\\Folders\\Downloads\\segnext_base_512x512_ade_160k.pth")
    # segNeXtLargeAttention = torch.load("E:\\Folders\\Downloads\\segnext_large_512x512_ade_160k.pth")
    # with open("E:\\Folders\\Downloads\\segnext_tiny_1024x1024_city_160k\\archive\\data.pkl", 'r') as f:
    #     data=pickle.load(f)
    #     print(data)

    # conv_spatial0 = nn.Conv2d(3, 3, 3, stride=1, padding=0, groups=3, dilation=2)

    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # inputs = []
    # inputs.append(torch.rand(16, 32, 128, 128))
    # inputs.append(torch.rand(16, 64, 64, 64))
    # inputs.append(torch.rand(16, 160, 32, 32))
    # inputs.append(torch.rand(16, 256, 16, 16))
    # segHeadB0 = SegUnetFormerHeadB0(in_index=[0, 1, 2, 3], feature_strides=[4, 8, 16, 32], in_channels=[32, 64, 160, 256],
    #                                 channels=128, num_classes=2, decoder_params=dict(embed_dim=256))
    # segHeadB0(inputs)

    # model = SegFormerB0()
    #inp = torch.rand(1, 3, 512, 512)
    #model = AttentionModuleEx2(3)
    #model(inp)
    # inp2 = torch.rand(10, 7, 512, 512)
    # _c = torch.cat([inp, inp2], dim=1)
    #inp = inp + inp2
    # a = inp.size()[2:]  dim, dim, (1, 7), padding=(0, 3), groups=dim, dilation=3

    # conv_spatial0 = nn.Conv2d(3, 3, 5, stride=1, padding=6, groups=3, dilation=3)  # 深度空洞卷积
    # conv_spatial1 = nn.Conv2d(3, 3, (1, 7), stride=1, padding=(0, 9), groups=3, dilation=3)  # 深度空洞卷积
    # #conv_spatial1 = nn.Conv2d(3, 3, 7, stride=1, padding=9, groups=3, dilation=3)  # 深度空洞卷积
    # conv_spatial2 = nn.Conv2d(3, 3, 11, stride=1, padding=15, groups=3, dilation=3)  # 深度空洞卷积
    # conv_spatial3 = nn.Conv2d(3, 3, 21, stride=1, padding=30, groups=3, dilation=3)  # 深度空洞卷积
    #
    # outp0 = conv_spatial0(inp)
    # outp1 = conv_spatial1(inp)
    # outp2 = conv_spatial2(inp)
    # outp3 = conv_spatial3(inp)
    # out = inp + outp1 + outp2 + outp3

    # import torch
    #
    # x = torch.arange(24).view(2, 3, 4).float()
    # y = x.mean(1)
    # print("x.shape:", x.shape)
    # print("x:")
    # print(x)
    # print("y.shape:", y.shape)
    # print("y:")
    # print(y)
    # mscan_test = torch.load("../data/pretrained_models/segNeXt/mscan_test.pth")
    # mscan_t = torch.load("../data/pretrained_models/segNeXt/mscan_t.pth")
    # mscanCheckPoint = torch.load("G:/AutoDLResult/20230627-103637-MSCAN_Tiny-224/model_best.pth")
    # dcanEx11CheckPoint = torch.load("G:/AutoDLResult/20230628-224223-DCAN_Tiny-224_EX11/model_best.pth")
    # dcanEx10CheckPoint = torch.load("G:/AutoDLResult/20230630-080710-DCAN_Tiny-224_Ex10/model_best.pth")
    # dcanEx9CheckPoint = torch.load("G:/AutoDLResult/20230701-174414-DCAN_Tiny-224_Ex9/model_best.pth")
    # dcanEx8CheckPoint = torch.load("G:/AutoDLResult/20230703-044706-DCAN_Tiny-224_EX8/model_best.pth")
    # dcanEx14CheckPoint = torch.load("G:/AutoDLResult/20230705-082522-DCAN_Tiny-224_Ex14/model_best.pth")
    # dcanEx15CheckPoint = torch.load("G:/AutoDLResult/20230706-171253-DCAN_Tiny-224_Ex15/model_best.pth")
    # mscanSynsBNCheckPoint = torch.load("G:/AutoDLResult/20230708-014339-MSCAN_Tiny-224_75.776_308/model_best.pth")

    # mit_b0 = torch.load("../data/pretrained_models/segNeXtTinyAttentionBackbone.pth")

    # latest = torch.load("../data/pretrained_models/latest.pth")
    # upernet_van_b2_512x512_160k_ade20k = torch.load("../data/pretrained_models/upernet_van_b2_512x512_160k_ade20k.pth")
    # segnext_large_1024x1024_city_160k = torch.load("../data/pretrained_models/segnext_large_1024x1024_city_160k.pth")
    print('###########################')

    # mit_b0 = torch.load("../data/pretrained_models/segformer/mit_b0.pth")
    #
    # array1 = [1, 2, 3, 5, 6]
    # C, H, W = 3, 4, 5
    # embedding = torch.randn(C, H, W)
    # layer_norm = nn.LayerNorm(W)
    # # Activate module
    # output = layer_norm(embedding)
    # print(output)
    #
    # # Image Example
    # input = embedding.unsqueeze(0)
    # # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # # as shown in the image below
    # layer_norm = nn.LayerNorm([C, H])
    # output = layer_norm(input)
    # print(output)



