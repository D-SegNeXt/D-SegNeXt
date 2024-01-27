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

    _parser.add_argument('--path', default='../data/deepglobe_1024_3', help='image path')  #deepglobe的voc数据集合
    _parser.add_argument('--out_path', default='../data/deepglobe_1024_3/out', help='image path') #输出目录

    _parser.add_argument('--sat_path', default='../data/deepglobe_1024_3/origin/train/', help='image path')  #原始数据的train
    _parser.add_argument('--mask_path', default='../data/deepglobe_1024_3/train/label/', help='image path')  # 原始数据的train

    _args = _parser.parse_args()
    _dateset_path = _args.path
    _out_path = _args.out_path
    _sat_path = _args.sat_path
    _mask_path = _args.mask_path

    # 保证目录存在
    _OutImageTrainSets = os.path.join(_out_path, 'train/')
    if (False == os.path.exists(_OutImageTrainSets)):
        os.makedirs(_OutImageTrainSets)

    with open(_dateset_path + '/ImageSets/Segmentation/train.txt', "r") as _trainFile:
        _trainNameList = _trainFile.read().splitlines()
        for _images in _trainNameList:
            _file_path = os.path.join(_sat_path, _images)
            _file_path += '_sat.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _OutImageTrainSets + _images
            _destFile += '_sat.png'
            shutil.copy(_file_path, _destFile)  # 复制文件

            _file_path = os.path.join(_sat_path, _images)
            _file_path += '_mask.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _OutImageTrainSets + _images
            _destFile += '_mask.png'
            shutil.copy(_file_path, _destFile)  # 复制文件


    _OutImageTestSets = os.path.join(_out_path, 'test/')
    if (False == os.path.exists(_OutImageTestSets)):
        os.makedirs(_OutImageTestSets)

    # with open(_dateset_path + '/ImageSets/Segmentation/test.txt', "r") as _testFile:
    #     _testNameList = _testFile.read().splitlines()
    #     for _images in _testNameList:
    #         _file_path = os.path.join(_sat_path, _images)
    #         _file_path += '_sat.png'
    #         if not os.path.isfile(_file_path):
    #             print("%s not exist!" % (_file_path))
    #
    #         _destFile = _OutImageTestSets + _images
    #         _destFile += '_sat.png'
    #         shutil.copy(_file_path, _destFile)  # 复制文件

            # _file_path = os.path.join(_sat_path, _images)
            # _file_path += '_mask.png'
            # if not os.path.isfile(_file_path):
            #     print("%s not exist!" % (_file_path))
            #
            # _destFile = _OutImageTestSets + _images
            # _destFile += '_mask.png'
            # shutil.copy(_file_path, _destFile)  # 复制文件


    _OutImageValidSets = os.path.join(_out_path, 'val/')
    if (False == os.path.exists(_OutImageValidSets)):
        os.makedirs(_OutImageValidSets)

    with open(_dateset_path + '/ImageSets/Segmentation/val.txt', "r") as _valFile:
        _valNameList = _valFile.read().splitlines()
        for _images in _valNameList:
            _file_path = os.path.join(_sat_path, _images)
            _file_path += '_sat.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _OutImageValidSets + _images
            _destFile += '_sat.png'
            shutil.copy(_file_path, _destFile)  # 复制文件

            _file_path = os.path.join(_sat_path, _images)
            _file_path += '_mask.png'
            if not os.path.isfile(_file_path):
                print("%s not exist!" % (_file_path))

            _destFile = _OutImageValidSets + _images
            _destFile += '_mask.png'
            shutil.copy(_file_path, _destFile)  # 复制文件
