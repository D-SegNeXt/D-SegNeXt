#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/11 15:11
# @Author  : FlyingRocCui
# @File    : splitTrainValTest.py
# @Description : 这里写注释内容

import os
import random  # 随机数包

import argparse
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='split the train、validation、testing ')

    _parser.add_argument('--imagepath', default='../data/OpenPitMine/ImageSets/', help='list path')
    _parser.add_argument('--path', default='../data/OpenPitMine/ImageSets/Segmentation/trainval.txt', help='list path')

    _args = _parser.parse_args()

    _trainValTxtFile = _args.path

    with open(_trainValTxtFile) as _trainValFile:
        _trainValNameList = _trainValFile.readlines()
        print("#####################################################################\n")
        print(len(_trainValNameList))
        print(_trainValNameList)
        train_list, val_list = train_test_split(_trainValNameList, test_size=0.4, random_state=12345)  # 按照比例分割训练集和验证集
        _trainTxtFile = _trainValTxtFile.replace("trainval", "train")
        with open(_trainTxtFile, "wt") as _trainFile:
            _trainFile.writelines(train_list)
            _trainFile.close()

        _valTxtFile = _trainValTxtFile.replace("trainval", "val")
        with open(_valTxtFile, "wt") as _valFile:
            _valFile.writelines(val_list)
            _valFile.close()



MainFolder = '../data/OpenPitMine/ImageSets/Segmentation/'
TrainValTestFiles = {'train': 'train.txt',
                     'val': 'val.txt',
                     'trainval': 'trainval.txt',
                     'test': 'test.txt'}  # 图片集划分文件集合
TrainR = 0.7  # 用于训练的数据量占比
ValR = 0.2  # 用于验证的数据量占比
PreImNum = 100  # 数据总量
fileIdLen = 6  # 图片名字字符数量，不够补0占位


def CreateImIdTxt(ImIdS, FilePath):
    if os.path.exists(FilePath):
        os.remove(FilePath)  # 保存的文件夹下有同名的文件先删除
    with open(FilePath, 'w') as FId:
        for ImId in ImIdS:
            ImIdStr = str(ImId).zfill(fileIdLen) + '\n'  # 占位换行
            FId.writelines(ImIdStr)


# ImIdSet = range(1, PreImNum + 1)  # 图片名标记从1开始
ImIdSet = [i for i in range(100)]
random.shuffle(ImIdSet)  # 随机打乱这个集合
ImNum = len(ImIdSet)
TrainNum = int(TrainR * ImNum)  # 用于训练的图片数量
ValNum = int(ValR * ImNum)  # 用于验证的图片数量

TrainImId = ImIdSet[:TrainNum - 1]  # 从打乱的集合中抽取前TrainNum个数据
TrainImId.sort()  # 从小到大排序，主要是为了好看
ValImId = ImIdSet[TrainNum:TrainNum + ValNum - 1]  # 从打乱的集合中抽取ValNum个数据
ValImId.sort()
TrainValImId = list(set(TrainImId).union(set(ValImId)))  # train和val集合组合成trainval
TrainValImId.sort()
TestImId = (list(set(ImIdSet).difference(set(TrainValImId))))  # 从总集合中除去trainval就是test
TestImId.sort()
TrainValTestIds = {}  # 把上述集合按字典方式组合在一起
TrainValTestIds['train'] = TrainImId
TrainValTestIds['val'] = ValImId
TrainValTestIds['trainval'] = TrainValImId
TrainValTestIds['test'] = TestImId

for Key, KeyVal in TrainValTestFiles.items():  # 遍历字典产生文件
    ImIdS = TrainValTestIds[Key]
    FileName = TrainValTestFiles[Key]
    FilePath = os.path.join(MainFolder, FileName)
    CreateImIdTxt(ImIdS, FilePath)
