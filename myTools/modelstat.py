#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 9:53
# @Author  : FlyingRocCui
# @File    : modelstat.py
# @Description : 模型信息统计
import torchvision.models
import torch
import torch.nn as nn
import torchsummary

from torchstat import stat

# Number of params: 342.00
class AttentionModule(nn.Module):
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
#Number of params: 342.00
class AttentionModuleEx1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)#聚合局部信息
        #以下为多分枝
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 9), groups=dim, dilation=3)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(9, 0), groups=dim, dilation=3)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 15), groups=dim, dilation=3)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(15, 0), groups=dim, dilation=3)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 30), groups=dim, dilation=3)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(30, 0), groups=dim, dilation=3)

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
# Number of params: 684.00
class AttentionModuleEx(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)#聚合局部信息
        #以下为多分枝
        self.conv_spatial0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 深度空洞卷积
        self.conv_spatial1 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # 深度空洞卷积
        self.conv_spatial2 = nn.Conv2d(dim, dim, 11, padding=5, groups=dim)  # 深度空洞卷积
        self.conv_spatial3 = nn.Conv2d(dim, dim, 21, padding=10, groups=dim)   # 深度空洞卷积

        # self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        #
        # self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        #
        # self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)#1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv_spatial0(attn)

        attn_1 = self.conv_spatial1(attn)

        attn_2 = self.conv_spatial2(attn)

        attn_3 = self.conv_spatial3(attn)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn * u
def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2f' % (total ))
#Number of params: 402.00
class AttentionModule4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)#聚合局部信息
        #以下为多分枝
        self.conv1_1 = nn.Conv2d(dim, dim, (5, 5), padding=(2, 2), groups=dim, dilation=1)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 5), padding=(4, 4), groups=dim, dilation=2)
        self.conv1_3 = nn.Conv2d(dim, dim, (5, 5), padding=(6, 6), groups=dim, dilation=3)
        self.conv1_4 = nn.Conv2d(dim, dim, (5, 5), padding=(8, 8), groups=dim, dilation=4)


        self.conv3 = nn.Conv2d(dim, dim, 1)#1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_1 = self.conv1_1(attn)
        attn_2 = self.conv1_2(attn_1)
        attn_3 = self.conv1_3(attn_2)
        attn_4 = self.conv1_4(attn_3)

        attn = attn + attn_3
        attn = self.conv3(attn)

        return attn * u
# Number of params: 324.00
class AttentionModule5(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 聚合局部信息
        # 以下为多分枝
        self.conv1_1 = nn.Conv2d(dim, dim, (5, 5), padding=(2, 2), groups=dim, dilation=1)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 5), padding=(4, 4), groups=dim, dilation=2)
        self.conv1_3 = nn.Conv2d(dim, dim, (5, 5), padding=(18, 18), groups=dim, dilation=9)

        self.conv3 = nn.Conv2d(dim, dim, 1)  # 1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_1 = self.conv1_1(attn)
        attn_2 = self.conv1_2(attn_1)
        attn_3 = self.conv1_3(attn_2)

        attn = attn + attn_3
        attn = self.conv3(attn)

        return attn * u
#Number of params: 540.00
class AttentionModule6(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)#聚合局部信息
        #以下为多分枝
        self.conv1_1 = nn.Conv2d(dim, dim, (7, 7), padding=(3, 3), groups=dim, dilation=1)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 7), padding=(6, 6), groups=dim, dilation=2)
        self.conv1_3 = nn.Conv2d(dim, dim, (7, 7), padding=(9, 9), groups=dim, dilation=3)
        #self.conv4 = nn.Conv2d(dim, dim, (7, 7), padding=(12, 12), groups=dim, dilation=4)


        self.conv4 = nn.Conv2d(dim, dim, 1)#1X1卷积建立不同分支之间的关系

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_1 = self.conv1_1(attn)
        attn_2 = self.conv1_2(attn_1)
        attn_3 = self.conv1_3(attn_2)
        #attn_4 = self.conv4(attn_3)

        attn = attn + attn_3
        attn = self.conv4(attn)

        return attn * u
#Number of params: 246.00
if __name__ == '__main__':
    # model = torchvision.models.vgg16(pretrained=False)
    model = AttentionModuleEx(4)
    device = torch.device('cpu')
    model.to(device)
    print_model_parm_nums(model)

    print('torchsummary++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    torchsummary.summary(model.cuda(), (4, 224, 224))
    print('stat++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    #stat(model.to(device), (3, 224, 224))

    # transformerModel = mit_b0()
    # stat(transformerModel.to(device), (1, 3, 512, 512))