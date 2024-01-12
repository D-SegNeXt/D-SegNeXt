#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/24 14:30
# @Author  : FlyingRocCui
# @File    : getBackbonePth.py
# @Description : 这里写注释内容
import torch
import argparse
import os
import shutil

if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description="'get the backbone's pameters, and save")

    _parser.add_argument('--inPthPath', default='../data/pretrained_models/segNeXt/mscan_s.pth', help='in pth path')
    _parser.add_argument('--outPthPath', default='../data/pretrained_models/segNeXt/mscan_s_BB.pth', help='out pth path')
    _parser.add_argument('--epoch', default=300, type=int, help='image path')
    _parser.add_argument('--version', default=1, type=int, help='image path')
    _parser.add_argument('--name', default='van_tiny_attention', help='image path')

    _args = _parser.parse_args()

    _out = {}
    _out['epoch'] = _args.epoch
    _out['name'] = _args.name
    _out['version'] = _args.version
    _inPthPath = _args.inPthPath
    _outPthPath = _args.outPthPath

    _outstate_dict = {}
    segNeXtTinyAttention = torch.load(_inPthPath)
    if 'state_dict' in segNeXtTinyAttention:
        _state_dict = segNeXtTinyAttention['state_dict']
        for key, value in _state_dict.items():
            if key.startswith('backbone.'):
                _outstate_dict[key[9:]] = value
            else:
                _outstate_dict[key] = value

    _out['state_dict'] = _outstate_dict
    torch.save(_out, _outPthPath)