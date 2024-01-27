#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 16:56
# @Author  : FlyingRocCui
# @File    : getImagesMeanStd.py
# @Description : 获取数据集的均值和标准差

import argparse
import os
import numpy as np
import cv2
from PIL import Image


#python ./mytools/getImagesMeanStd.py
if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='get the mean and std of the images')

    _parser.add_argument('--path', default='../data/deepglobe_1024_3/JPEGImages', help='image path')
    _parser.add_argument('--munChannel', default=3, type=int, help='the channel of the image')

    _args = _parser.parse_args()
    _imagesPath = _args.path
    _numChannel = _args.munChannel

    _means = np.zeros((_numChannel,), dtype=float)
    _std = np.zeros((_numChannel,), dtype=float)
    _ImgCount = 0

    for root, dirs, files in os.walk(_imagesPath):
        if root != _imagesPath:
            break
        for file in files:
            _path = os.path.join(root, file)
            if not os.path.isfile(_path):
                print("%s not exist!" % (_path))
            else:
                _ImgCount += 1
                print(f"Images No {_ImgCount}")
                _ImgData = Image.open(_path)
                _ImgData2 = np.array(_ImgData)
                for d in range(_numChannel):
                    _means[d] += _ImgData2[:, :, d].mean()
                    _std[d] += _ImgData2[:, :, d].std()

    _means = np.array(_means) / _ImgCount
    _std =  np.array(_std) / _ImgCount
    print(f"mean: {_means}")
    print(f"std: {_std}")