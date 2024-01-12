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

### ADE20K

|   Method  |    Backbone     |  Pretrained | Iters | mIoU(ss/ms) | Params | FLOPs  | Config | Download  |
| :-------: | :-------------: | :-----: | :---: | :--: | :----: | :----: | :----: | :-------: |
|  SegNeXt  |     MSCAN-T  | IN-1K | 160K | 41.1/42.2 | 4M | 7G | [config](local_configs/segnext/tiny/segnext.tiny.512x512.ade.160k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/5da98841b8384ba0988a/?dl=1) |
|  SegNeXt  |     MSCAN-S | IN-1K  | 160K |  44.3/45.8  | 14M | 16G | [config](local_configs/segnext/small/segnext.small.512x512.ade.160k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/b2d1eb94f5944d60b3d2/?dl=1) |
|  SegNeXt  |     MSCAN-B  | IN-1K  | 160K |  48.5/49.9 | 28M | 35G | [config](local_configs/segnext/base/segnext.base.512x512.ade.160k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/1ea8000916284493810b/?dl=1) |
|  SegNeXt  |     MSCAN-L  | IN-1K  | 160K |  51.0/52.1 | 49M | 70G | [config](local_configs/segnext/large/segnext.large.512x512.ade.160k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/d4f8e1020643414fbf7f/?dl=1) |


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