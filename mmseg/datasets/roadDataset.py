#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/6 13:55
# @Author  : FlyingRocCui
# @File    : RoadDataset.py
# @Description : 这里写注释内容


import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()  # 注册   不要忘记在__init__.py作显示导入
class RoadDataset(CustomDataset):
    CLASSES = ('background', 'road')   # 类别名称设置
    PALETTE = [[0, 0, 0], [128, 0, 0]]  # 调色板设置

    def __init__(self,**kwargs):
        super(RoadDataset, self).__init__(
            img_suffix='.png',  # img文件‘后缀’
            seg_map_suffix='.png',  # gt文件‘后缀’

            # """
            #    对于二分类设成False,对于多分类，视数据集而定，对于ade20k为True
            #    因为0代表背景，但是不包含在150个类别中
            # """
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)