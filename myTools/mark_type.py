#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 8:24
# @Author  : FlyingRocCui
# @File    : mark_type.py
# @Description : 这里写注释内容
import cv2 as cv
import numpy as np

def mark_type(infer,gt,ori_path,save_path):
    print(ori_path)
    ori=cv.imread(ori_path)
    #print(ori.shape)
    mark_r = ori[:,:,2]
    mark_g = ori[:,:,1]
    mark_b = ori[:,:,0]
    #print(mark_r.shape)

    tp_matrix=np.multiply(infer,gt)#真正例，标记为白色
    fp_matrix=np.multiply(infer,1-gt)#假正例，标记为绿色
    tn_matrix = np.multiply(1-infer, 1 - gt)#真反例，标记为黑色
    fn_matrix = np.multiply(1-infer,  gt)#假反例，标记为红色
    #True Positive
    mark_r[np.where(tp_matrix == 1)]=255    #White
    mark_g[np.where(tp_matrix == 1)]=255
    mark_b[np.where(tp_matrix == 1)]=255
    #False Positive
    mark_g[np.where(fp_matrix == 1)] = 255  #Green
    # False Negatives
    mark_r[np.where(fn_matrix == 1)] = 255  #Red

    mark=np.stack((mark_b,mark_g,mark_r),axis=2)

    cv.imwrite(save_path,mark)
    print(save_path)


ori_dir="D:/doctor/data/OpenPitMine/JPEGImages/"  #原始文件
gt_path="D:/doctor/data/OpenPitMine/SegmentationClass/"     #标注文件
inferred_path="D:/doctor/logs/backup/compare/20230919_092115_DinkNet34_6e-5/"  #推测文件
mark_dst_path="D:/doctor/logs/backup/compare/D-LinkNet_compare_result2/"   #保存路径
pic_list_path="D:/doctor/data/OpenPitMine/ImageSets/Segmentation/test.txt" #文件列表，不含后缀名
inferred_list=[]
gt_list=[]
with open(pic_list_path,'r') as f:
    lines = f.readlines()
    for line in lines:
        inferred=inferred_path+line.strip("\n")+"_sat_mask.png"
        gt=gt_path+line.strip("\n")+".png"
        # infer_arr=cv.imread(inferred,cv.IMREAD_GRAYSCALE)
        # gt_arr=cv.imread(gt,cv.IMREAD_GRAYSCALE)
        # infer_arr[infer_arr==255]=1
        # gt_arr[gt_arr==255]=1
        infer_arr = cv.imread(inferred)
        gt_arr = cv.imread(gt)
        # print(ori.shape)
        infer_arr_r = infer_arr[:, :, 2]
        infer_arr_r[infer_arr_r > 120] = 1

        gt_arr_r = gt_arr[:, :, 2]
        gt_arr_r[gt_arr_r > 120] = 1

        save_path=mark_dst_path+line.strip("\n")+".png"
        ori_path=ori_dir+line.strip("\n")+".png"
        mark_type(infer_arr_r,gt_arr_r,ori_path,save_path)
