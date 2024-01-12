#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 13:52
# @Author  : FlyingRocCui
# @File    : mineImage.py
# @Description : 准备露天矿的图片数据
import argparse
import os
import numpy as np
import cv2
from PIL import Image


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='resize the image of the open pit mine')

    _parser.add_argument('--path', default='../data/shenbao/Split/raster', help='image path')
    _parser.add_argument('--outpath', default='../data/shenbao/Split/rasterResult', help='image out path')
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
        _ImgData = _ImgData.resize((_image_size, _image_size), Image.Resampling.BILINEAR)

        if _mode == 0:
            _ImgData = _ImgData.point(lambda i: 0 if i == 0 else 1)
            _ImgData.info.clear()
            _ImgData.putpalette(_paletteStand)
        else:
            _ImgData = _ImgData.convert('RGB')
            _ImgData.info = {}
        # print("rgbImg 模式:", _ImgData.mode)
        # print("rgbImg 尺寸:", _ImgData.size)
        # print("rgbImg 通道数:", len(_ImgData.split()))

        outFile = os.path.join(_image_outpath, _namelist[i])
        _ImgData.save(outFile)

