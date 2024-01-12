# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer
from .segUnetFormer import *
from .van import *
from .unet2 import *
from .linknet34 import LinkNet34

# from .mix_transformer import *   #by FRC 2023-03-03
from .mscan import *   #by FRC 2023-05-11
from .dcan import *   #by FRC 2023-07-3
from .DLinkNet import *
from .dlinknet2 import DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from .dunet import Dunet
from .nl_linknet34 import NL34_LinkNet
__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone', 'ERFNet', 'PCPVT',
    'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE', 'SegFormer',
    'MSCAN', 'DCAN','DLinkNet','VAN',
    'UNet2', 'LinkNet34', 'Dunet', 'NL34_LinkNet',
    'DinkNet34', 'DinkNet50', 'DinkNet101', 'DinkNet34_less_pool'
]
