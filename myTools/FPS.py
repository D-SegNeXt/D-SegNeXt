#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/26 14:44
# @Author  : FlyingRocCui
# @File    : FPS.py
# @Description : 这里写注释内容

import torch
import torch.utils.data as data
import os
import warnings
from time import *
import mmseg
import numpy as np
import time

if __name__ == '__main__':
    model = DCAN_Tiny().cuda()

    times = []
    for i in range(200):
        x1 = torch.randn(1, 3, 1024, 1024).cuda()

        start = time.time()
        predict = model(x1)
        end = time.time()
        times.append(end - start)

    print(f"FPS: {1.0 / np.mean(times):.3f}")