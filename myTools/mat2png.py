#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/10 16:46
# @Author  : FlyingRocCui
# @File    : mat2png.py
# @Description : 这里写注释内容
import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import mat73

# 数据矩阵转图片的函数
def MatrixToImage(data):
    #data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='convert mat to png')

    _parser.add_argument('--path', default='../data/coco_stuff10k/annotations', help='the mat file path')
    _parser.add_argument('--outPath', default='../data/coco_stuff10k/annotationsPng', help='the mat file path')

    _args = _parser.parse_args()
    _matFilespath = _args.path
    _outPath = _args.outPath
    _ImgCount = 0

    for root, dirs, files in os.walk(_matFilespath):
        if root != _matFilespath:
            break
        for file in files:
            _path = os.path.join(root, file)
            if not os.path.isfile(_path):
                print("%s not exist!" % (_path))
            else:
                _ImgCount += 1
                print(f"Images No {_ImgCount}")

                # mat = mat73.loadmat(_path)
                # new_name = str(mat.keys())
                # key_name = list(mat.keys())[-1]
                # key_name = mat[key_name]
                # print(key_name.shape)

                array_struct = scio.loadmat(_path)
                # print(array_struct)
                # 校验步骤
                array_data = array_struct['S']  # 取出需要的数字矩阵部分
                # print(array_data)
                # 校验步骤
                new_im = MatrixToImage(array_data)  # 调用函数
                #plt.imshow(array_data, cmap=plt.cm.gray, interpolation='nearest')
                # new_im.show()
                # print(first_name)
                _outFilepath = os.path.join(_outPath, file)
                new_im.save(_outFilepath.replace(".mat", "_labelTrainIds.png"))  # 保存图片
