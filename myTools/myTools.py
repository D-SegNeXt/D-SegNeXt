#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/6 8:09
# @Author  : FlyingRocCui
# @File    : divideTrainValidTest.py
# @Description : 这里写注释内容

import dataProcessor
import dataSetProcessor

# command_dict = {
#      "pl2csv":1,
#      "exit":-1
# }

class Function: #结构体
    def __init__(self, description, function):
        self.m_Description = description
        self.m_Function = function

command_dict = {
     1 : Function("pl2csv", "dataProcessor.pkl2csv('./dataset/Beijing/GPS/beijing_gps_dir_speed_interval_sorted.pkl', './dataset/Beijing/GPS/beijing_gps_dir_speed_interval_sorted.csv')"),
     2 : Function("timeFormat", "dataProcessor.timeFormat()"),
     3 : Function("testFunction", "dataProcessor.testFunction()"),
     4 : Function("stayPoint", "dataProcessor.stayPoint('../data/shenbao/20220620-20220703_no3.csv', '../data/shenbao/20220620-20220703_no3Result50.csv')"),
     5 : Function("mergePng",  "dataProcessor.mergePng('../data/CMMPNET/BJRoad/test/image/')"),
     6 : Function("convertVoc",  "dataProcessor.convertVoc('../data/CMMPNET/BJRoad/', '../data/CMMPNETVOC_1024_4/')"),
     7 : Function("voc2cityscapes",  "dataSetProcessor.voc2cityscapes('../data/CMMPNETVOC/SegmentationClass/', '../data/CMMPNETVOC/SegmentationClass2/')"),
     8 : Function("resizeImage",  "dataProcessor.resizeImage('../data/CMMPNETVOC4/SegmentationClass/')"),
     9 : Function("loadImage",  "dataProcessor.loadImage('../data/CMMPNETVOC/SegmentationClass/2_17_merge.png')"),
     10 : Function("loadImage",  "dataProcessor.loadImage('../data/VOC2012/SegmentationClass/2007_000033.png')"),
     11 : Function("divideTrainValidTest",  "dataProcessor.divideTrainValidTest('')"),
     12 : Function("getMeanStd",  "dataProcessor.getMeanStd('../data/CMMPNETVOC_1024_4/JPEGImages')"),
     13 : Function("changeRGB2Label",  "dataProcessor.changeRGB2Label('../data/CMMPNETVOC_1024_4/SegmentationClass')"),
     14 : Function("change2Png",  "dataProcessor.change2Png('../data/CMMPNETVOC_1024_3/JPEGImages')"),
     0 : Function("exit",'exit()')
}

if __name__ == '__main__':
    for key in command_dict:
        print(key, ":", command_dict[key].m_Description)

    while True:
        input_id = int(input('Please input the command ID:'))
        print('Your command id is ', input_id)
        if(input_id in command_dict):
            eval(command_dict[input_id].m_Function)
        else:
            print('can not find the id', ':', input_id)
            continue