#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 9:17
# @Author  : FlyingRocCui
# @File    : deepglobe2voc.py
# @Description : 将deepgloberoad 数据集转换纬voc形式
import argparse
import os
import numpy as np
import cv2
from PIL import Image
import shutil

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='convert deepglobe 2 voc')

    _parser.add_argument('--path', default='../data/deepglobe_1024_3', help='image path')

    _args = _parser.parse_args()
    _dateset_path = _args.path

    # 保证目录存在
    _ImageSets = os.path.join(_dateset_path, 'ImageSets/Segmentation/')
    if (False == os.path.exists(_ImageSets)):
        os.makedirs(_ImageSets)

    _JPEGImages = os.path.join(_dateset_path, 'JPEGImages/')
    if (False == os.path.exists(_JPEGImages)):
        os.makedirs(_JPEGImages)

    _SegmentationClass = os.path.join(_dateset_path, 'SegmentationClass/')
    if (False == os.path.exists(_SegmentationClass)):
        os.makedirs(_SegmentationClass)

    _SegmentationObject = os.path.join(_dateset_path, 'SegmentationObject/')
    if (False == os.path.exists(_SegmentationObject)):
        os.makedirs(_SegmentationObject)

    # 拷贝train文件
    _train_image = os.path.join(_dateset_path, 'train/img')
    _valid_image = os.path.join(_dateset_path, 'valid/img')
    with open(_ImageSets + 'train.txt', "wt") as _trainFile:
        with open(_ImageSets + 'trainval.txt', "wt") as _trainValFile:
            for root, dirs, files in os.walk(_train_image):
                if root != _train_image:
                    break
                for file in files:
                    _path = os.path.join(root, file)
                    if not os.path.isfile(_path):
                        print("%s not exist!" % (_path))
                    else:
                        _trainFile.writelines(file.split("_")[0] + '\n')
                        _trainValFile.writelines(file.split("_")[0] + '\n')
                        _destFile = _JPEGImages + file
                        _destFile = _destFile.replace("_sat.jpg", ".jpg")
                        shutil.copy(_path, _destFile)  # 复制文件

                for root, dirs, files in os.walk(_valid_image):
                    if root != _valid_image:
                        break
                    for file in files:
                        _path = os.path.join(root, file)
                        if not os.path.isfile(_path):
                            print("%s not exist!" % (_path))
                        else:
                            _trainValFile.writelines(file.split("_")[0] + '\n')
                            _destFile = _JPEGImages + file
                            _destFile = _destFile.replace("_sat.jpg", ".jpg")
                            shutil.copy(_path, _destFile)  # 复制文件
        _trainValFile.close()
    _trainFile.close()

    _train_label = os.path.join(_dateset_path, 'train/label')
    for root, dirs, files in os.walk(_train_label):
        if root != _train_label:
            break
        for file in files:
            _path = os.path.join(root, file)
            if not os.path.isfile(_path):
                print("%s not exist!" % (_path))
            else:
                _destFile = _SegmentationClass + file
                _destFile = _destFile.replace("_mask.png", ".png")
                shutil.copy(_path, _destFile)  # 复制文件

    _valid_label = os.path.join(_dateset_path, 'valid/label')
    for root, dirs, files in os.walk(_valid_label):
        if root != _valid_label:
            break
        for file in files:
            _path = os.path.join(root, file)
            if not os.path.isfile(_path):
                print("%s not exist!" % (_path))
            else:
                _destFile = _SegmentationClass + file
                _destFile = _destFile.replace("_mask.png", ".png")
                shutil.copy(_path, _destFile)  # 复制文件

    # 拷贝test文件
    _test_image = os.path.join(_dateset_path, 'test/img')
    with open(_ImageSets + '/test.txt', "wt") as _testFile:
        for root, dirs, files in os.walk(_test_image):
            if root != _test_image:
                break
            for file in files:
                _path = os.path.join(root, file)
                if not os.path.isfile(_path):
                    print("%s not exist!" % (_path))
                else:
                    _testFile.writelines(file.split("_")[0] + '\n')
                    _destFile = _JPEGImages + file
                    _destFile = _destFile.replace("_sat.jpg", ".jpg")
                    shutil.copy(_path, _destFile)  # 复制文件
    _testFile.close()