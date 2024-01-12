# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
from beforeimport import *

from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmseg.models import build_segmentor



def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', default='./local_configs/segUnetFormer/B0/b0.512x512.cmmpnetvoc.160k.py', help='train config file path')
    # parser.add_argument('--config', default='./local_configs/segformer/B0/segformer.b0.512x512.cmmpnetvoc.160k.py', help='train config file path')

    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    # for _ in sys.path:
    #     print(_)
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])  # 通道数
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()


#python ./tools/get_flops.py ./myconfigs/segNeXt/tiny/segnext.tiny.ss_header_512x512.coco_stuff164k.80k.py
#python ./tools/get_flops.py ./myconfigs/dsegNeXt/tiny/dsegnext.tiny.512x512.OPM.20k.py
