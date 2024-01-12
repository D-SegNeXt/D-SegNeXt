#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 10:48
# @Author  : FlyingRocCui
# @File    : voc2deepglobe.py
# @Description : 将voc转换为deepglobe数据
import argparse
import os
import numpy as np
import cv2
from PIL import Image
import shutil

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='convert voc  2 deepglobe')

    _parser.add_argument('--path', default='../data/OpenPitMine', help='image path')
    _parser.add_argument('--out_path', default='../data/OpenPitMine_DP', help='image path')

    _parser.add_argument('--sat_path', default='E:/WorkRoom/Sources/doctor/data/shenbao/Split/rasterResult', help='image path')
    _parser.add_argument('--mask_path', default='E:/WorkRoom/Sources/doctor/data/shenbao/Split/shapeResult2', help='image path')

    _args = _parser.parse_args()
    _dateset_path = _args.path
    _out_path = _args.out_path
    _sat_path = _args.sat_path
    _mask_path = _args.mask_path

    # 保证目录存在
    _ImageTrainSets = os.path.join(_out_path, 'train/')
    if (False == os.path.exists(_ImageTrainSets)):
        os.makedirs(_ImageTrainSets)

    _segmentation_path = os.path.join(_dateset_path, 'ImageSets/Segmentation')
    with open(_dateset_path + '/ImageSets/Segmentation/train.txt', "r") as _trainFile:
        _trainNameList = _trainFile.read().splitlines()
        for _images in _trainNameList:
            _file_path = os.path.join(_sat_path, _images)
            _file_path += '.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _ImageTrainSets + _images
            _destFile += '_sat.png'
            shutil.copy(_file_path, _destFile)  # 复制文件

            _file_path = os.path.join(_mask_path, _images)
            _file_path += '.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _ImageTrainSets + _images
            _destFile += '_mask.png'
            shutil.copy(_file_path, _destFile)  # 复制文件


    _ImageTestSets = os.path.join(_out_path, 'test/')
    if (False == os.path.exists(_ImageTestSets)):
        os.makedirs(_ImageTestSets)

    with open(_dateset_path + '/ImageSets/Segmentation/test.txt', "r") as _testFile:
        _testNameList = _testFile.read().splitlines()
        for _images in _testNameList:
            _file_path = os.path.join(_sat_path, _images)
            _file_path += '.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _ImageTestSets + _images
            _destFile += '_sat.png'
            shutil.copy(_file_path, _destFile)  # 复制文件

            _file_path = os.path.join(_mask_path, _images)
            _file_path += '.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _ImageTestSets + _images
            _destFile += '_mask.png'
            shutil.copy(_file_path, _destFile)  # 复制文件


    _ImageValidSets = os.path.join(_out_path, 'val/')
    if (False == os.path.exists(_ImageValidSets)):
        os.makedirs(_ImageValidSets)

    with open(_dateset_path + '/ImageSets/Segmentation/val.txt', "r") as _valFile:
        _valNameList = _valFile.read().splitlines()
        for _images in _testNameList:
            _file_path = os.path.join(_sat_path, _images)
            _file_path += '.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _ImageValidSets + _images
            _destFile += '_sat.png'
            shutil.copy(_file_path, _destFile)  # 复制文件

            _file_path = os.path.join(_mask_path, _images)
            _file_path += '.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _ImageValidSets + _images
            _destFile += '_mask.png'
            shutil.copy(_file_path, _destFile)  # 复制文件
