#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/22 16:39
# @Author  : FlyingRocCui
# @File    : dataSetProcessor.py
# @Description : 这里写注释内容
import os
from PIL import Image


def voc2cityscapes(inpath, outpath):
    if (False == os.path.exists(outpath)):
        os.makedirs(outpath)

    for root, dirs, files in os.walk(inpath):
        if root != inpath:
            break
        for file in files:
            path = os.path.join(root, file)
            alphaImg = Image.open(path)
            alphaImg = alphaImg.point(lambda i: 128 if i > 0 else 0)
            print("alphaImg模式:", alphaImg.mode)
            print("alphaImg尺寸:", alphaImg.size)
            print("alphaImg通道数:", len(alphaImg.split()))

            outFile = os.path.join(outpath, file)
            alphaImg.save(outFile)