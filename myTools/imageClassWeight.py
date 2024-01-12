#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 13:58
# @Author  : FlyingRocCui
# @File    : imageClassWeight.py
# @Description : 计算图像分类中各类的比例
import argparse
import os
import numpy as np
import cv2
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='caculate the class weight of the labels')

    parser.add_argument('--path', default='../data/CMMPNETVOC_1024_3/SegmentationClass/',
                        help='test image path')
    parser.add_argument('--class_no', default=2, type=int, help='the number of the class')

    args = parser.parse_args()

    _pixel_count = np.zeros((args.class_no, 1))
    _image_path = args.path
    _namelist = os.listdir(_image_path)
    for i in range(len(_namelist)):
        _imageFile = os.path.join(_image_path, _namelist[i])
        print(i, _imageFile)
        # label = cv2.imread(imageFile, 0)
        _ImgData = Image.open(_imageFile)
        _ImgData2 = np.array(_ImgData)

        _label_uni = np.unique(_ImgData2)
        for m in _label_uni:
            _pixel_count[m] += np.sum(_ImgData2 == m) / _ImgData2.size

    W = 1 / np.log(_pixel_count.T)
    W = args.class_no * W / np.sum(W)
    print(W)