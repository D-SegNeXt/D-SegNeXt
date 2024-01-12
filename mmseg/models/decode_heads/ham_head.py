import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=None,
            act_cfg=None
        )

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham

class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)
        # self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1, padding=0, groups=inplanes)
        # self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        # x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

@HEADS.register_module()
class LightHamHead(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 **kwargs):
        super(LightHamHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)
        #self.hamburger = DUpsampling(self.ham_channels, 2, self.ham_channels)
        #self.hamburger = DUC(self.ham_channels, self.ham_channels * 4, 2)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',#bilinear
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output

class SpatialSelectionModule(nn.Module):
    def __init__(self):
        super(SpatialSelectionModule, self).__init__()
        # self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.silu = nn.SiLU()
        # self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))

    def forward(self, x):
        atten = self.silu(x.mean(dim=1).unsqueeze(dim=1))
        feat = torch.mul(x, atten)
        feat = x + feat
        return feat

@HEADS.register_module()
class LightSSHead(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 **kwargs):
        super(LightSSHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.spatialSelection = SpatialSelectionModule()
        #self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)
        #self.hamburger = DUpsampling(self.ham_channels, 2, self.ham_channels)
        #self.hamburger = DUC(self.ham_channels, self.ham_channels * 4, 2)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inout0 = inputs[0]
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inout0.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.spatialSelection(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output

class SpatialSelectionModule2(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=None,
            act_cfg=None
        )

        #self.ham = nn.SpatialSelectionModule()
        #self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)

        atten = self.relu(enjoy.mean(dim=1).unsqueeze(dim=1))
        feat = torch.mul(enjoy, atten)
        enjoy = enjoy + feat

        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham
@HEADS.register_module()
class LightSSHead2(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO:
        Add other MD models (Ham).
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 **kwargs):
        super(LightSSHead2, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.spatialSelection = SpatialSelectionModule2()
        #self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)
        # self.hamburger = DUpsampling(self.ham_channels, 2, self.ham_channels)
        # self.hamburger = DUC(self.ham_channels, self.ham_channels * 4, 2)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inout0 = inputs[0]
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inout0.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        #x = self.spatialSelection(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output

@HEADS.register_module()
class LightHamHeadEx(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO:
        Add other MD models (Ham).
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 **kwargs):
        super(LightHamHeadEx, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        for i in range(len(self.in_channels)):
            if i == 0:
                squeeze = ConvModule(
                    self.in_channels[0],
                    self.ham_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            else:
                squeeze = ConvModule(
                    self.in_channels[i] + self.ham_channels,
                    self.ham_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

            setattr(self, f"squeeze{i + 1}", squeeze)
            setattr(self, f"hamburger{i + 1}", Hamburger(ham_channels, ham_kwargs, **kwargs))

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]
        _mPre = None
        for i in range(len(inputs)):
            squeeze = getattr(self, f"squeeze{i + 1}")
            hamburger = getattr(self, f"hamburger{i + 1}")
            if i == 0:
                _mPre = squeeze(inputs[i])
                _mPre = hamburger(_mPre)
            else:
                _mPre = torch.cat([_mPre, inputs[i]], dim=1)
                _mPre = squeeze(_mPre)
                _mPre = hamburger(_mPre)

        output = self.align(_mPre)
        output = self.cls_seg(output)
        return output

class DUCK3(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUCK3, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        # self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1, padding=0, groups=inplanes)
        # self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        # x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)

        return x

@HEADS.register_module()
#DUCHead4 利用DUpsampling对每个层进行预测，然后融合，然后再预测
class DUCHead(BaseDecodeHead):
    def __init__(self,
                 up_channels=[32, 128, 512, 2048],
                 **kwargs):
        super(DUCHead, self).__init__(input_transform='multiple_select', **kwargs)

        _InChannel0 = self.in_channels[self.in_index[0]]
        self.DUC1 = DUpsampling(_InChannel0, 8, self.num_classes)

        _InChannel1 = self.in_channels[self.in_index[1]]
        self.DUC2 = DUpsampling(_InChannel1, 4, self.num_classes)

        _InChannel2 = self.in_channels[self.in_index[2]]
        self.DUC3 = DUpsampling(_InChannel2, 2, self.num_classes)

        self.squeeze = ConvModule(
            self.num_classes * 3,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.linear_fuse = ConvModule(
            in_channels= self.channels,
            out_channels= self.channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        _out1 = self.DUC1(inputs[0])
        _out2 = self.DUC2(inputs[1])
        _out3 = self.DUC3(inputs[2])
        #_out4 = inputs[3]
        _out = torch.cat([_out1, _out2, _out3], dim=1)
        output = self.squeeze(_out)
        output = self.linear_fuse(output)
        output = self.cls_seg(output)
        return output

# DUCHead3
#倒数第12层分辨率分别变为第三层并与之融合进行预测，卷积核为3
# class DUCHead3(BaseDecodeHead):
#     def __init__(self,
#                  up_channels=[32, 128, 512, 2048],
#                  **kwargs):
#         super(DUCHead, self).__init__(input_transform='multiple_select', **kwargs)
#
#         _InChannel0 = self.in_channels[self.in_index[0]]
#         self.DUC1 = DUCK3(_InChannel0, _InChannel0 * 16, 4)
#
#         _InChannel1 = self.in_channels[self.in_index[1]]
#         self.DUC2 = DUCK3(_InChannel1,  _InChannel1 * 4, 2)
#
#         _InChannel2 = self.in_channels[self.in_index[2]]
#         self.DUC3 = DUCK3(_InChannel0 + _InChannel1 + _InChannel2, (_InChannel0 + _InChannel1 + _InChannel2) * 4, 2)
#
#         self.squeeze = ConvModule(
#             _InChannel0 + _InChannel1 + _InChannel2,
#             self.channels,
#             1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#
#         self.linear_fuse = ConvModule(
#             in_channels= self.channels,
#             out_channels= self.channels,
#             kernel_size=1,
#             norm_cfg=dict(type='BN', requires_grad=True)
#         )
#
#     def forward(self, inputs):
#         """Forward function."""
#         inputs = self._transform_inputs(inputs)
#
#         _out1 = self.DUC1(inputs[0])
#         _out2 = self.DUC2(inputs[1])
#         _out3 = self.DUC3(torch.cat([_out1, _out2, inputs[2]], dim=1))
#         output = self.squeeze(_out3)
#         output = self.linear_fuse(output)
#         output = self.cls_seg(output)
#         return output

# DUCHead2
#下层分辨率扩大后与上层进行融合一直到正数第二层
# class DUCHead2(BaseDecodeHead):
#     def __init__(self,
#                  up_channels=[32, 128, 512, 2048],
#                  **kwargs):
#         super(DUCHead2, self).__init__(input_transform='multiple_select', **kwargs)
#
#         _InChannel0 = self.in_channels[self.in_index[0]]
#         self.DUC1 = DUC(_InChannel0, _InChannel0 * 4, 2)
#
#         _InChannel1 = self.in_channels[self.in_index[1]]
#         self.DUC2 = DUC(_InChannel0 + _InChannel1,  (_InChannel0 + _InChannel1) * 4, 2)
#
#         _InChannel2 = self.in_channels[self.in_index[2]]
#         self.DUC3 = DUC(_InChannel0 + _InChannel1 + _InChannel2, (_InChannel0 + _InChannel1 + _InChannel2) * 4, 2)
#
#         self.squeeze = ConvModule(
#             _InChannel0 + _InChannel1 + _InChannel2,
#             self.channels,
#             1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#
#         self.linear_fuse = ConvModule(
#             in_channels= self.channels,
#             out_channels= self.channels,
#             kernel_size=1,
#             norm_cfg=dict(type='BN', requires_grad=True)
#         )
#
#     def forward(self, inputs):
#         """Forward function."""
#         inputs = self._transform_inputs(inputs)
#
#         _out1 = self.DUC1(inputs[0])
#         _out2 = self.DUC2(torch.cat([_out1, inputs[1]], dim=1))
#         _out3 = self.DUC3(torch.cat([_out2, inputs[2]], dim=1))
#         #_out4 = inputs[3]
#         #_out = torch.cat([_out3, _out4], dim=1)
#         output = self.squeeze(_out3)
#         output = self.linear_fuse(output)
#         output = self.cls_seg(output)
#         return output

#DUCHead1
#倒数第12层分辨率分别变为第三层并与之融合进行预测
# class DUCHead1(BaseDecodeHead):
#     def __init__(self,
#                  up_channels=[32, 128, 512, 2048],
#                  **kwargs):
#         super(DUCHead1, self).__init__(input_transform='multiple_select', **kwargs)
#
#         _InChannel0 = self.in_channels[self.in_index[0]]
#         self.DUC1 = DUC(_InChannel0, _InChannel0 * 16, 4)
#
#         _InChannel1 = self.in_channels[self.in_index[1]]
#         self.DUC2 = DUC(_InChannel1,  _InChannel1 * 4, 2)
#
#         _InChannel2 = self.in_channels[self.in_index[2]]
#         self.DUC3 = DUC(_InChannel0 + _InChannel1 + _InChannel2, (_InChannel0 + _InChannel1 + _InChannel2) * 4, 2)
#
#         self.squeeze = ConvModule(
#             _InChannel0 + _InChannel1 + _InChannel2,
#             self.channels,
#             1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#
#         self.linear_fuse = ConvModule(
#             in_channels= self.channels,
#             out_channels= self.channels,
#             kernel_size=1,
#             norm_cfg=dict(type='BN', requires_grad=True)
#         )
#
#     def forward(self, inputs):
#         """Forward function."""
#         inputs = self._transform_inputs(inputs)
#
#         _out1 = self.DUC1(inputs[0])
#         _out2 = self.DUC2(inputs[1])
#         _out3 = self.DUC3(torch.cat([_out1, _out2, inputs[2]], dim=1))
#         output = self.squeeze(_out3)
#         output = self.linear_fuse(output)
#         output = self.cls_seg(output)
#         return output
###########################################################################
# class DUCHead(BaseDecodeHead):
#     def __init__(self,
#                  up_channels=[32, 128, 512, 2048],
#                  **kwargs):
#         super(DUCHead, self).__init__(input_transform='multiple_select', **kwargs)
#
#         # for i in range(len(self.in_index)):
#         #     _IndexID = self.in_index[i]
#         #     _DUC = DUC(self.in_channels[_IndexID - 1], up_channels[i]);
#         #
#         #     setattr(self, f"DUC{i + 1}", _DUC)
#         _InChannel0 = self.in_channels[self.in_index[0]]
#         self.DUC1 = DUC(_InChannel0, _InChannel0 * 4, 2)
#
#         _InChannel1 = self.in_channels[self.in_index[1]]
#         self.DUC2 = DUC(_InChannel0 ,  _InChannel0 * 4, 2)
#
#         _InChannel2 = self.in_channels[self.in_index[2]]
#         self.DUC3 = DUC(_InChannel0 ,  _InChannel0 * 4, 2)
#
#         self.squeeze = ConvModule(
#             _InChannel0,
#             self.channels,
#             1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#
#         self.linear_fuse = ConvModule(
#             in_channels= self.channels,
#             out_channels= self.channels,
#             kernel_size=1,
#             norm_cfg=dict(type='BN', requires_grad=True)
#         )
#
#     def forward(self, inputs):
#         """Forward function."""
#         inputs = self._transform_inputs(inputs)
#
#         # _mOut = None
#         # for i in range(len(self.in_index)):
#         #     _IndexID = self.in_index[i]
#         #     _DUC = getattr(self, f"DUC{i + 1}")
#         #     if i == 0:
#         #         _mOut = inputs[i]
#         #     else:
#         #         _mOut += inputs[i]
#         #
#         #     _mOut = _DUC(_mOut)
#         #
#         # _mOut = self.cls_seg(_mOut)
#         # return _mOut
#         _out1 = self.DUC1(inputs[0])
#         _out2 = self.DUC2(_out1)
#         _out3 = self.DUC3(_out2)
#         #_out4 = inputs[3]
#         #_out = torch.cat([_out3, _out4], dim=1)
#         output = self.squeeze(_out3)
#         output = self.linear_fuse(output)
#         output = self.cls_seg(output)
#         return output