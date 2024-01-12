# Road Extraction from High-Resolution Remote Sensing Images of Open-Pit Mine Using D-SegNeXt

The repository contains official Pytorch implementations of training and evaluation codes models for **D-SegNext**. 

The paper will be available in the future.

The code is based on [MMSegmentaion v0.30](https://github.com/open-mmlab/mmsegmentation/tree/v0.30).


## Citation
If you find our repo useful for your research, please consider citing our paper:

```

will update in future

```

## Results

**Notes**: ImageNet-1k Pre-trained models can be found in [Baidu Cloud](https://pan.baidu.com/s/1qE18p7Zg1iYjq9rWl9OJTQ?pwd=omtq).

## DataSet
A high-quality open-pit mine road dataset The dataset can be found in [Baidu Cloud](https://pan.baidu.com/s/1YN9lky921LUYWy2be1gsOg).

### OPM

|   Method  |    Backbone     |  Pretrained | Iters | mIoU | Params | FLOPs  | Config | Download  |
| :-------: | :-------------: | :-----: | :---: | :--: | :----: | :----: | :----: | :-------: |
|  D-SegNeXt  |     MSCAN-T  | IN-1K | 20K | 41.1 | 5.3M | 6.8G | [config](myconfigs/dsegNeXt/tiny/dsegnext.tiny.1024x1024.OPM.20k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1X7Y1RNbtvr6uUsXSZ_r7iA?pwd=6gnh) |
|  D-SegNeXt  |     MSCAN-S | IN-1K  | 20K |  44.3  | 17.3M | 17.76G | [config](myconfigs/dsegNeXt/small/dsegnext.small.1024x1024.OPM.20k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1n4NK-0joBiUxV0vZT9qjFg?pwd=a39k) |
|  D-SegNeXt  |     MSCAN-B  | IN-1K  | 40K |  48.5 | 34.2M | 37.56G | [config](myconfigs/dsegNeXt/base/dsegnext.base.1024x1024.OPM.40k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1WqMkca_h7UvqO_lG8hZI8Q?pwd=wgkx) |


**Notes**: In this scheme, The number of FLOPs (G) is calculated on the input size of 512 $\times$ 512.



## Installation
Install the dependencies  according to the guidelines in [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

## Training

We use 2 RTX5000 GPUs for training by default. Run:

```bash
bash ./tools/dist_train.sh './myconfigs/dsegNeXt/tiny/dsegnext.tiny.1024x1024.OPM.20k.py' 2
```

## Evaluation

To evaluate the model, run:

```bash
bash ./tools/dist_test.sh '../logs/20230913_140414/segnext.tiny.512x512.OPM.20k.py' '../logs/20230913_140414/latest.pth' 2 --show-dir='../logs/20230913_140414/result/'
./tools/dist_test.sh /path/to/config /path/to/checkpoint_file 8 --eval mIoU
```

## FLOPs

Install torchprofile using

```bash
pip install torchprofile
```

To calculate FLOPs for a model, run:

```bash
bash tools/get_flops.py /path/to/config --shape 512 512
```

## Contact

If you have any private question, please feel free to contact us via flyingroccui@sina.com.

## Acknowledgment

Our implementation is mainly based on [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt/tree/main) and [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.30). Thanks for their authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.