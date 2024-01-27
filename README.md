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

|   Method  | Pretrained | Iters | mIoU | Params | FLOPs  | Config | Download  |
| :-------: | :---: | :--: | :----: | :----: | :----: | :-------: |
|  LinkNet  | -- | 20K | 71.45 | 21.64M | 27.38G |  |  |
| D-LinkNet | -- | 20K | 71.75 | 31.09M | 23.59G |  |  |
|    UNet   | -- | 20K | 59.29 | 39.50M | 8.74G |  |  |
|   DUnet   | -- | 20K | 72.53 | 19.20M | 282.46G |  |  |
| NL-LinkNet | -- | 20K | 71.69 | 21.82M | 31.44G |  |  |
| :-------: | :-----: | :---: | :--: | :----: | :----: | :----: | :-------: |
|Segformer-b0| IN-1K | 20K | 72.24 | 3.72M | 6.36G |[config](myconfigs/segformer/segformer.b0.1024x1024.OPM.20k.py)|  |
| SegNeXt-T| IN-1K | 20K | 71.52 | 4.23M | 6.04G |[config](myconfigs/segnext/tiny/segnext.tiny.1024x1024.OPM.20k.py)|  |
|D-SegNeXt-T| None | 20K | 61.14 | 5.30M | 6.82G |[config](myconfigs/dsegnext/tiny/Dsegnext.tiny.1024x1024.OPM.20k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1X7Y1RNbtvr6uUsXSZ_r7iA?pwd=6gnh) |
|D-SegNeXt-T|  IN-1K | 20K | 73.32 | 5.30M | 6.82G |[config](myconfigs/dsegnext/tiny/Dsegnext.tiny.1024x1024.OPM.20k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1X7Y1RNbtvr6uUsXSZ_r7iA?pwd=6gnh) |
| :-------: | :-----: | :---: | :--: | :----: | :----: | :----: | :-------: |
|Segformer-b1| IN-1K | 20K | 74.02 | 13.68M | 11.63G |[config](myconfigs/segformer/segformer.b1.1024x1024.OPM.20k.py)|  |
|SegNeXt-S| IN-1K | 20K | 74.24 | 13.89M | 15.32G |[config](myconfigs/segnext/small/segnext.small.1024x1024.OPM.20k.py)|  |
|D-SegNeXt-S| None | 20K | 68.36 | 17.32M | 17.76G | [config](myconfigs/dsegnext/small/dsegnext.small.1024x1024.OPM.20k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1n4NK-0joBiUxV0vZT9qjFg?pwd=a39k) |
|D-SegNeXt-S| IN-1K | 20K | 74.61 | 17.32M | 17.76G | [config](myconfigs/dsegnext/small/dsegnext.small.1024x1024.OPM.20k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1n4NK-0joBiUxV0vZT9qjFg?pwd=a39k) |
| :-------: | :-----: | :---: | :--: | :----: | :----: | :----: | :-------: |
|Segformer-b2| IN-1K | 20K | 75.42 | 24.72M | 17.93G |[config](myconfigs/segformer/segformer.b2.1024x1024.OPM.40k.py)|  |
| SegNeXt-B | IN-1K | 20K | 75.41 | 27.56M | 32.02G |[config](myconfigs/segnext/base/segnext.base.1024x1024.OPM.40k.py)|  |
|D-SegNeXt-B| IN-1K | 20K | 72.12 | 34.27M | 37.56G | [config](myconfigs/dsegnext/tiny/Dsegnext.tiny.1024x1024.OPM.20k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1X7Y1RNbtvr6uUsXSZ_r7iA?pwd=6gnh) |
|D-SegNeXt-B|  IN-1K  | 40K |  75.59 | 34.27M | 37.56G | [config](myconfigs/dsegnext/base/dsegnext.base.1024x1024.OPM.40k.py)  | [Baidu Cloud](https://pan.baidu.com/s/1WqMkca_h7UvqO_lG8hZI8Q?pwd=wgkx) |

### DeepGlobe Road Extraction

|   Method  | Pretrained | Iters | mIoU | Params | FLOPs  | Config | 
| :-------: | :--------: | :---: | :--: | :----: | :----: | :----: |
|  LinkNet  | -- | 20K | 62.12 | 21.64M | 27.38G |  |
| D-LinkNet | -- | 20K | 64.69 | 31.09M | 23.59G |  |
|    UNet   | -- | 20K | 54.37 | 39.50M | 8.74G |  |
|   DUnet   | -- | 20K | 65.23 | 19.20M | 282.46G |  |
| NL-LinkNet | --| 20K | 64.63 | 21.82M | 31.44G |  |
| :-------: | :--------: | :---: | :--: | :----: | :----: | :----: |
|Segformer-b0| IN-1K | 20K | 72.24 | 3.72M | 6.36G |[config](myconfigs/segformer/segformer.b0.1024x1024.DP.20k.py)|
| SegNeXt-T| IN-1K | 20K | 71.52 | 4.23M | 6.04G |[config](myconfigs/segnext/tiny/segnext.tiny.1024x1024.DP.20k.py)|
|D-SegNeXt-T| None | 20K | 61.14 | 5.30M | 6.82G |[config](myconfigs/dsegnext/tiny/Dsegnext.tiny.1024x1024.DP.20k.py)|
|D-SegNeXt-T|  IN-1K | 20K | 73.32 | 5.30M | 6.82G |[config](myconfigs/dsegnext/tiny/Dsegnext.tiny.1024x1024.DP.20k.py)|
| :-------: | :--------: | :---: | :--: | :----: | :----: | :----: |
|Segformer-b1| IN-1K | 20K | 64.76 | 13.68M | 11.63G |[config](myconfigs/segformer/segformer.b1.1024x1024.DP.20k.py)|
|SegNeXt-S| IN-1K | 20K | 65.65 | 13.89M | 15.32G |[config](myconfigs/segnext/small/segnext.small.1024x1024.DP.20k.py)|
|D-SegNeXt-S| None | 20K | 62.69 | 17.32M | 17.76G | [config](myconfigs/dsegnext/small/dsegnext.small.1024x1024.DP.20k.py)|
|D-SegNeXt-S| IN-1K | 20K | 67.39 | 17.32M | 17.76G | [config](myconfigs/dsegnext/small/dsegnext.small.1024x1024.DP.20k.py)|
| :-------: | :--------: | :---: | :--: | :----: | :----: | :----: |
|Segformer-b2| IN-1K | 20K | 66.78 | 24.72M | 17.93G |[config](myconfigs/segformer/segformer.b2.1024x1024.DP.40k.py)|
| SegNeXt-B | IN-1K | 20K | 67.92 | 27.56M | 32.02G |[config](myconfigs/segnext/base/segnext.base.1024x1024.DP.40k.py)|
|D-SegNeXt-B| IN-1K | 20K | 63.53 | 34.27M | 37.56G | [config](myconfigs/dsegnext/tiny/Dsegnext.tiny.1024x1024.OPM.20k.py)|
|D-SegNeXt-B|  IN-1K  | 40K | 67.96 | 34.27M | 37.56G | [config](myconfigs/dsegnext/base/dsegnext.base.1024x1024.OPM.40k.py)|

**Notes**: In this scheme, The number of FLOPs (G) is calculated on the input size of 512 $\times$ 512.

## Ablation Study

Ablation study on the design of D-SegNeXt_base. IoU demotes IoU on OPM benchmark. Acc demotes accuracy on OPM benchmark.

|           Method          | Params | FLOPs | mIoU | Acc  |
| :-----------------------: | :-----: | :---: | :--: | :----: | 
| D-SegNeXt-B w/o Attention | 34.27M | 37.56G | 65.73 | 82.12 |
| D-SegNeXt-B w/o Dilated Conv | 34.27M | 37.56G | 70.23 | 83.56 |
| D-SegNeXt-B w/o Residual connection | 34.27M | 37.56G |65.64|81.23|
| D-SegNeXt-B w/o Multi-Scales | 25.49M | 29.39G | 70.17 | 84.41 |
| D-SegNeXt-B w/o Above All | 34.27M | 37.56G |60.46|77.36|
| D-SegNeXt-B w/o Pretrain | 34.27M | 37.56G | 72.12 | 85.13 |
| D-SegNeXt-B  | 34.27M | 37.56G | 75.59 | 87.31 |

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