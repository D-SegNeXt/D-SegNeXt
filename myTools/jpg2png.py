#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 7:40
# @Author  : FlyingRocCui
# @File    : jpg2png.py
# @Description : 将文件格式转换为png,分辨率等其他信息不变

import argparse
import os
import numpy as np
import cv2
from PIL import Image


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='jpg to png')

    _parser.add_argument('--path', default='../data/CMMPNETVOC512_512_3/JPEGImages', help='image path')

    _args = _parser.parse_args()

    _image_path = _args.path
    _namelist = os.listdir(_image_path)

    for i in range(len(_namelist)):
        _imageFile = os.path.join(_image_path, _namelist[i])
        print(i, _imageFile)
        _ImgData = Image.open(_imageFile)

        _destFile = _imageFile.replace(".jpg", ".png")
        _ImgData.save(_destFile)


