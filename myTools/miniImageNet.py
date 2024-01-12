#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/7 7:47
# @Author  : FlyingRocCui
# @File    : miniImageNet.py
# @Description : 抽取ImageNet1K生成新的数据集，可控制train和validation中每个种类的数量
import argparse
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='Mini the ImageNet')

    _parser.add_argument('--imageNet1Kpath', default='E:/Data/ImageNet/images', help='image path')
    _parser.add_argument('--outpath', default='E:/Data/MiniImageNet/images', help='image out path')
    _parser.add_argument('--trainSize', default=130, type=int, help='the size of the train')
    _parser.add_argument('--validationSize', default=5, type=int, help='the size of the train')

    _args = _parser.parse_args()

    _imageNet1Kpath = _args.imageNet1Kpath
    _outpath = _args.outpath
    _trainSize = _args.trainSize
    _validationSize = _args.validationSize
    _imageNet1KTrainPath = _imageNet1Kpath + '/train'
    _imageNet1KValidationPath = _imageNet1Kpath + '/validation'

    # for root, dirs, files in os.walk(_imageNet1KTrainPath):
    #     if root != _imageNet1KTrainPath:
    #         break
    #     for root1, dirs1, files1 in os.walk(root):
    #         for dir in dirs1:
    #             root2, _, filenames = next(os.walk(root1 + '/' + dir))
    #             _testPer = _trainSize / len(filenames)
    #             trainval_list, test_list = train_test_split(filenames, test_size=_testPer, random_state=12345)
    #             for filename in test_list:
    #                 _filePath = root2 + '/' + filename
    #                 _dist = _outpath + '/train/' + dir
    #                 folder = os.path.exists(_dist)
    #
    #                 if not folder:
    #                     os.makedirs(_dist)
    #                 shutil.copy(_filePath, _outpath + '/train/' + dir + '/' + filename)

    for root, dirs, files in os.walk(_imageNet1KValidationPath):
        if root != _imageNet1KValidationPath:
            break
        for root1, dirs1, files1 in os.walk(root):
            for dir in dirs1:
                root2, _, filenames = next(os.walk(root1 + '/' + dir))
                _testPer = _validationSize / len(filenames)
                trainval_list, test_list = train_test_split(filenames, test_size=_testPer, random_state=12345)
                for filename in test_list:
                    _filePath = root2 + '/' + filename
                    _dist = _outpath + '/validation/' + dir
                    folder = os.path.exists(_dist)
                    if not folder:
                        os.makedirs(_dist)
                    shutil.copy(_filePath, _outpath + '/validation/' + dir + '/' + filename)
