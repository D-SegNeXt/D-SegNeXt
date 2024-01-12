#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/19 12:33
# @Author  : FlyingRocCui
# @File    : PSSL2VOC.py
# @Description : 将PSSL数据转换为VOC格式
import argparse
import os
import numpy as np
import cv2
from PIL import Image
import shutil

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='get the mean and std of the images')

    _parser.add_argument('--path', default='D:/pssl2.1_consensus/train/', help='image path')
    _parser.add_argument('--label', default='D:/pssl2.1_consensus/label/', help='image path')

    _parser.add_argument('--imageSet', default='D:/pssl2.1_consensus/imageSet/', help='image path')
    _parser.add_argument('--imagespath',
                         default='F:/Download/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/',
                         help='image path')

    _args = _parser.parse_args()
    _InPath = _args.path
    _outPath = _args.label
    _imagespath = _args.imagespath
    _imageSet = _args.imageSet
    #生成调色板

    imageStand = Image.open('../data/VOC2012/SegmentationClass/2007_000033.png')
    paletteStand = imageStand.getpalette()

    _myPalette = []
    _myPalette.append(0)
    _myPalette.append(0)
    _myPalette.append(0)
    for i in range(1, 11):
        for j in range(1, 11):
            for k in range(1, 11):
                _myPalette.append(25 * i)
                _myPalette.append(25 * j)
                _myPalette.append(25 * k)

    _classID = 0
    for _root0, _dirs0, _files0 in os.walk(_InPath):
        if _root0 != _InPath:
            break
        for _dir0 in _dirs0:
            _wholeDir = os.path.join(_root0, _dir0)
            print(_wholeDir)
            _classID += 1
            if _classID > 256:
                continue
            for _root1, _dirs1, _files1 in os.walk(_wholeDir):
                for _file1 in _files1:
                    _path = os.path.join(_root1, _file1)
                    if not os.path.isfile(_path):
                        print("%s not exist!" % (_path))
                    else:
                        _outFile = _file1.replace(".JPEG_eiseg.npz", ".png")
                        _outFile = os.path.join(_outPath, _outFile)
                        _npz = np.load(_path)
                        _npzlabel = _npz['arr_0']
                        _npzlabelOut = np.zeros_like(_npzlabel, dtype=np.int64)
                        # # [0, 999] for imagenet classes, 1000 for background, others(-1) will be ignored during training.
                        _npzlabelOut[_npzlabel == 1] = _classID
                        _image = Image.fromarray(np.ushort(_npzlabelOut))
                        _image = _image.convert('P')
                        _image.putpalette(paletteStand)
                        _image.save(_outFile)

                        _fromFile = os.path.join(_imagespath, _dir0, _file1.replace(".JPEG_eiseg.npz", ".jpeg"))
                        shutil.copy(_fromFile, os.path.join(_imageSet, _file1.replace(".JPEG_eiseg.npz", ".jpg")))  # 复制文件
