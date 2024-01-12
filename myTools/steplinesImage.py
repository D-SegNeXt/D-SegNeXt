#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 13:52
# @Author  : FlyingRocCui
# @File    : mineImage.py
# @Description : 处理坡线数据
import argparse
import os

import PIL.ImageOps
import numpy as np
import cv2
from PIL import Image


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='resize the image of the open pit mine')

    _parser.add_argument('--path', default='D:/grid3/DOM_Out', help='image path')
    _parser.add_argument('--outpath', default='D:/grid3/DOM_Out2', help='image out path')
    _parser.add_argument('--size', default=1024, type=int, help='the size of the image')
    _parser.add_argument('--mode', default=1, type=int, help='0:label 1:image')

    _imageStand = Image.open('../data/VOC2012/SegmentationClass/2007_000033.png')
    _paletteStand = _imageStand.getpalette()

    _args = _parser.parse_args()

    _image_path = _args.path
    _image_outpath = _args.outpath
    _image_size = _args.size
    _namelist = os.listdir(_image_path)
    _mode = _args.mode

    for i in range(len(_namelist)):
        _imageFile = os.path.join(_image_path, _namelist[i])
        print(i, _imageFile)
        _ImgData = Image.open(_imageFile)
        #_ImgData = _ImgData.resize((_image_size, _image_size), Image.Resampling.BILINEAR)

        if _mode == 0:
            # aa = _ImgData.getbands()
            _ImgData = _ImgData.convert('L')
            #image1 = np.array(_ImgData)
            _ImgData = _ImgData.point(lambda i: 0 if i == 255 else 1)
            #image2 = np.array(_ImgData)
            _ImgData.info.clear()
            _ImgData.putpalette(_paletteStand)
            # bb = _ImgData.getbands()
        else:
            _ImgData = _ImgData.convert('RGB')
            _ImgData.info = {}

        outFile = os.path.join(_image_outpath, _namelist[i])
        _ImgData.save(outFile)

