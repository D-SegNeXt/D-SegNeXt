#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/2 16:31
# @Author  : FlyingRocCui
# @File    : dataProcessor.py
# @Description : 这里写注释内容


import pickle
import pandas as pd
import  time
from PIL import Image
import os
import shutil
import numpy as np

#pkl转换为csv格式
def pkl2csv(infile, outfile):
    print('pkl2csv')
    pklFile = open(infile, 'rb')
    pklFile = pickle.load(pklFile, encoding='iso-8859-1')
    pdDataFrame = pd.DataFrame(pklFile)
    print(pdDataFrame.columns)
    pdDataFrame["time2"] = pdDataFrame.time.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x))))
    pdDataFrame.to_csv(outfile, index=False, encoding='utf-8')

def timeFormat():
    timeStamp = 1228061046

    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    print(otherStyleTime)



    import datetime
    timeStamp = 1228061046
    dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
    otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
    print(otherStyleTime)

def testFunction():
    dates = ['April-20', 'April-21', 'April-22', 'April-23', 'April-24', 'April-25']
    income = [1228097636, 1228097954, 1228098278, 1228098278, 1228098278, 1228098278]
    expenses = [3, 8, 4, 5, 6, 10]

    df = pd.DataFrame({"Date": dates, "time": income, "Expenses": expenses})
    df["time2"] = df.time.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    print(df)

def stayPoint(infile, outfile):
    pdDataFrame = pd.read_csv(infile, dtype={"equ_id": str,"equ_code": str})
    pdDataFrame.sort_values(by  = ['equ_id','dt'])
    pdDataFrame['equ_id2'] = pdDataFrame['equ_id'].shift(-1)
    pdDataFrame['dt2'] = pdDataFrame['dt'].shift(-1)
    pdDataFrame['st_x2'] = pdDataFrame['st_x'].shift(-1)
    pdDataFrame['st_y2'] = pdDataFrame['st_y'].shift(-1)
    pdDataFrame['stayPoint'] = ((pdDataFrame['equ_id2'] == pdDataFrame['equ_id']) &
    (abs(pdDataFrame['st_x2'] - pdDataFrame['st_x']) < 5) &
    (abs(pdDataFrame['st_y2'] - pdDataFrame['st_y']) < 5))

    print(pdDataFrame.shape[0])

    index_names = pdDataFrame[pdDataFrame['stayPoint'] == True].index
    pdDataFrame.drop(index_names, inplace=True)

    print(pdDataFrame.shape[0])
    pdDataFrame.to_csv(outfile, index=False, encoding='utf-8')

def mergePng(rgbpath):
    mergepath = rgbpath.replace("image", "merge")
    if(False == os.path.exists(mergepath)):
        os.makedirs(mergepath)

    for root, dirs, files in os.walk(rgbpath):
        if root != rgbpath:
            break
        for file in files:
            path = os.path.join(root, file)
            rgbImg = Image.open(path)
            print("rgbImg 模式:", rgbImg.mode)
            print("rgbImg 尺寸:", rgbImg.size)
            print("rgbImg 通道数:", len(rgbImg.split()))
            rChannel, gChannel, bChannel = rgbImg.split()

            alphaFile = path.replace("sat.png", "gps.jpg")
            alphaFile = alphaFile.replace("image", "gps")
            alphaImg = Image.open(alphaFile)
            alphaImg = alphaImg.point(lambda _: 255 - _)
            print("alphaImg模式:", alphaImg.mode)
            print("alphaImg尺寸:", alphaImg.size)
            print("alphaImg通道数:", len(alphaImg.split()))
            if(rgbImg.size != alphaImg.size):
                continue

            aChannel = alphaImg.split()
            #image_rgba = Image.merge("RGB", (rChannel, gChannel, bChannel))
            image_rgba = Image.merge("RGBA", (rChannel, gChannel, bChannel, aChannel[0]))
            #image_rgba = image_rgba.resize((512, 512), Image.BILINEAR)

            mergeFile = os.path.join(mergepath, file)
            mergeFile = mergeFile.replace("sat.png", "merge.png")
            image_rgba.save(mergeFile)

def convertVoc(datasetPath, vocPath):
    #创建voc的目录
    ImageSets = os.path.join(vocPath, 'ImageSets/Segmentation/')
    if (False == os.path.exists(ImageSets)):
        os.makedirs(ImageSets)

    IPEGImages = os.path.join(vocPath, 'JPEGImages/')
    if (False == os.path.exists(IPEGImages)):
        os.makedirs(IPEGImages)

    SegmentationClass = os.path.join(vocPath, 'SegmentationClass/')
    if (False == os.path.exists(SegmentationClass)):
        os.makedirs(SegmentationClass)

    SegmentationObject = os.path.join(vocPath, 'SegmentationObject/')
    if (False == os.path.exists(SegmentationObject)):
        os.makedirs(SegmentationObject)

    #拷贝train文件
    train_valmerge = os.path.join(datasetPath, 'train_val/merge')
    with open(ImageSets + 'train.txt', "wt") as trainFile:
        with open(ImageSets + 'trainval.txt', "wt") as trainValFile:
            for root, dirs, files in os.walk(train_valmerge):
                if root != train_valmerge:
                    break
                for file in files:
                    path = os.path.join(root, file)
                    if not os.path.isfile(path):
                        print("%s not exist!" % (path))
                    else:
                        trainFile.writelines(file.split(".")[0] + '\n')
                        trainValFile.writelines(file.split(".")[0] + '\n')
                        shutil.copy(path, IPEGImages + file)  # 复制文件

    # 拷贝test文件
    testmerge = os.path.join(datasetPath, 'test/merge')
    with open(ImageSets + '/test.txt', "wt") as testFile:
        for root, dirs, files in os.walk(testmerge):
            if root != testmerge:
                break
            for file in files:
                path = os.path.join(root, file)
                if not os.path.isfile(path):
                    print("%s not exist!" % (path))
                else:
                    testFile.writelines(file.split(".")[0] + '\n')
                    shutil.copy(path, IPEGImages + file)  # 复制文件

    valFile = open(ImageSets + '/val.txt', "wt")
    valFile.close()

    # 拷贝mask数据
    maskmerge = os.path.join(datasetPath, 'test/mask')
    for root, dirs, files in os.walk(maskmerge):
        if root != maskmerge:
            break
        for file in files:
            path = os.path.join(root, file)
            if not os.path.isfile(path):
                print("%s not exist!" % (path))
            else:
                destFile = SegmentationClass + file
                destFile = destFile.replace("mask", "merge")
                shutil.copy(path, destFile)  # 复制文件

    maskmerge = os.path.join(datasetPath, 'train_val/mask')
    for root, dirs, files in os.walk(maskmerge):
        if root != maskmerge:
            break
        for file in files:
            path = os.path.join(root, file)
            if not os.path.isfile(path):
                print("%s not exist!" % (path))
            else:
                destFile = SegmentationClass + file
                destFile = destFile.replace("mask", "merge")
                shutil.copy(path, destFile)  # 复制文件

def resizeImage(filePath):
    imageStand = Image.open('../data/VOC2012/SegmentationClass/2007_000033.png')
    paletteStand = imageStand.getpalette()
    for root, dirs, files in os.walk(filePath):
        if root != filePath:
            break
        for file in files:
            path = os.path.join(root, file)
            imageOrign = Image.open(path)
            # imageOrign = imageOrign.resize((512, 512), Image.BILINEAR)
            imageOrign = imageOrign.convert('P')
            imageResize = imageOrign.point(lambda i: 0 if i == 0 else 1)
            imageResize.putpalette(paletteStand)
            # imageResize.putpalette([0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128,128, 0, 128, 0, 128, 128, 128,
            #                         128, 128, 64, 0, 0,192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64,
            #                         128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128])
            #
            imageResize.save(path)

def loadImage(filePath):
    imageOrign = Image.open(filePath)
    a = 1

    # imageOrign.show('Orign')
    #
    # imageResize =  imageOrign.resize((512, 512), Image.ANTIALIAS)
    # imageResize.show('ANTIALIAS')
    #
    # imageResize =  imageOrign.resize((512, 512), Image.NEAREST)
    # imageResize.show('NEAREST')
    #
    #
    # imageResize.show('BILINEAR')
    #
    # imageResize =  imageOrign.resize((512, 512), Image.ANTIALIAS)
    # imageResize.show('ANTIALIAS')

#获取目录下图片的均值和方差，默认为3个通道
def getMeanStd(dataSetPath):
    # calculate means and std  注意换行\n符号**
    # train.txt中每一行是图像的位置信息**
    _means = [0, 0, 0, 0]
    _std = [0, 0, 0, 0]
    _ImgCount = 0
    for root, dirs, files in os.walk(dataSetPath):
        if root != dataSetPath:
            break
        for file in files:
            _path = os.path.join(root, file)
            if not os.path.isfile(_path):
                print("%s not exist!" % (_path))
            else:
                _ImgCount += 1
                print(f"Images No {_ImgCount}")
                _ImgData = Image.open(_path)
                _ImgData2 = np.array(_ImgData)
                for d in range(4):
                    _means[d] += _ImgData2[:, :, d].mean()
                    _std[d] += _ImgData2[:, :, d].std()

    _means = np.array(_means) / _ImgCount
    _std =  np.array(_std) / _ImgCount
    print(f"mean: {_means}")
    print(f"std: {_std}")

def changeRGB2Label(filePath):
    imageStand = Image.open('../data/VOC2012/SegmentationClass/2007_000033.png')
    paletteStand = imageStand.getpalette()
    for root, dirs, files in os.walk(filePath):
        if root != filePath:
            break
        for file in files:
            path = os.path.join(root, file)
            imageOrign = Image.open(path)
            if imageOrign.mode == 'P':
                continue
            # imageOrign = imageOrign.resize((512, 512), Image.BILINEAR)
            imageOrign = imageOrign.convert('P')
            imageResize = imageOrign.point(lambda i: 0 if i == 0 else 1)
            imageResize.putpalette(paletteStand)
            # imageResize.putpalette([0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128,128, 0, 128, 0, 128, 128, 128,
            #                         128, 128, 64, 0, 0,192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64,
            #                         128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128])
            #
            imageResize.save(path)

def change2Png(filePath):
    for root, dirs, files in os.walk(filePath):
        if root != filePath:
            break
        for file in files:
            _path = os.path.join(root, file)
            _imageOrign = Image.open(_path)
            _destFile = _path.replace(".jpg", ".png")
            _imageOrign.save(_destFile)