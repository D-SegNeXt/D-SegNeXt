#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 13:08
# @Author  : FlyingRocCui
# @File    : splitTrainVal.py
# @Description : 这里写注释内容


import argparse
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

#dir /b >>txt.txt
if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='split the train and validation ')

    _parser.add_argument('--path', default='D:/grid3/trainvaltest.txt', help='list path')

    #_parser.add_argument('--path', default='../data/deepglobe_1024_3/ImageSets/Segmentation/trainvaltest.txt', help='list path')
    _trainPer = 0.6
    _valPer = 0.2
    _testPer = 0.2
    _args = _parser.parse_args()

    _trainValTestTxtFile = _args.path

    with open(_trainValTestTxtFile) as _trainValTestFile:
        _trainValTestNameList = _trainValTestFile.readlines()
        _trainValTestNameList.sort()
        print("#####################################################################\n")
        print(len(_trainValTestNameList))
        #print(_trainValNameList)
        trainval_list, test_list = train_test_split(_trainValTestNameList, test_size=_testPer, random_state=12345)  # 按照比例分割训练集和验证集
        _trainvalTxtFile = _trainValTestTxtFile.replace("trainvaltest", "trainval")
        with open(_trainvalTxtFile, "wt") as _trainValFile:
            _trainValFile.writelines(trainval_list)
            _trainValFile.close()

        _testTxtFile = _trainValTestTxtFile.replace("trainvaltest", "test")
        with open(_testTxtFile, "wt") as _testFile:
            _testFile.writelines(test_list)
            _testFile.close()

        train_list, val_list = train_test_split(trainval_list, test_size=_valPer / (_trainPer + _valPer), random_state=5)  # 按照比例分割训练集和验证集
        _trainTxtFile = _trainValTestTxtFile.replace("trainvaltest", "train")
        with open(_trainTxtFile, "wt") as _trainFile:
            _trainFile.writelines(train_list)
            _trainFile.close()

        _valTxtFile = _trainValTestTxtFile.replace("trainvaltest", "val")
        with open(_valTxtFile, "wt") as _valFile:
            _valFile.writelines(val_list)
            _valFile.close()
