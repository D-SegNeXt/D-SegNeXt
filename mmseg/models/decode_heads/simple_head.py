# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SimpleHead(BaseDecodeHead):
    def __init__(self,
                 **kwargs):
        super(SimpleHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.cls_seg(x)
        return output
