#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 15:55
# @Author  : FlyingRocCui
# @File    : radarReader.py
# @Description : 这里写注释内容
import struct

if __name__ == '__main__':
    with open('d:/train.txt', "wt") as ResultFile:
        with open('d:/2023_05_01_01_09_06.radar', 'rb') as originFile:
            pntNumber = struct.unpack("<i", originFile.read(4))[0]
            ResultFile.writelines(str(pntNumber) + '\n')
            for i in range(1, pntNumber + 1):
                double_value = struct.unpack("<d", originFile.read(8))[0]
                ResultFile.writelines(str(double_value) + ';')
                double_value = struct.unpack("<d", originFile.read(8))[0]
                ResultFile.writelines(str(double_value) + ';')
                double_value = struct.unpack("<d", originFile.read(8))[0]
                ResultFile.writelines(str(double_value) + '\n')
        originFile.close()
    ResultFile.close()
    # with open('d:/train2.txt', "wt") as ResultFile:
    #     with open('d:/2023_07_24_15_46_13.DiffImage', 'rb') as originFile:
    #         Row = struct.unpack("<i", originFile.read(4))[0]
    #         ResultFile.writelines(str(Row) + ';')
    #         Column = struct.unpack("<i", originFile.read(4))[0]
    #         ResultFile.writelines(str(Column) + '\n')
    #         for i in range(1, Row + Column + 1):
    #             double_value = struct.unpack("<d", originFile.read(8))[0]
    #             ResultFile.writelines(str(double_value) + '\n')
    #
    #         Column = struct.unpack("<i", originFile.read(4))[0]
    #         ResultFile.writelines(str(Column) + '\n')
    #     originFile.close()
    # ResultFile.close()