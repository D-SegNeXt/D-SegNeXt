#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/19 12:33
# @Author  : FlyingRocCui
# @File    : PSSL2VOC.py
# @Description : 检查ImageNet中数据后缀名shi
import argparse
import os
import shutil

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='get the mean and std of the images')

    _parser.add_argument('--path', default='E:/WorkRoom/Sources/doctor/data/ImageNet/train', help='image path')


    _args = _parser.parse_args()
    _InPath = _args.path


    for _root0, _dirs0, _files0 in os.walk(_InPath):
        if _root0 != _InPath:
            break
        for _dir0 in _dirs0:
            _wholeDir = os.path.join(_root0, _dir0)
            print(_dir0)
            for _root1, _dirs1, _files1 in os.walk(_wholeDir):
                for _file1 in _files1:
                    _path = os.path.join(_root1, _file1)

                    if not os.path.isfile(_path):
                        print("%s not exist!" % (_path))
                    else:
                        _extendx = _file1.split('.')[1]

                        if _extendx.lower() != 'jpeg':
                            print(_file1)


    print('end')