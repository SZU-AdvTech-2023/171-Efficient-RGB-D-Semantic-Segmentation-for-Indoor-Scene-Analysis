## Setup

1. Clone repository:
    ```bash
    git clone https://github.com/TUI-NICR/ESANet.git
   
    cd /path/to/this/repository
    ```

2. Set up anaconda environment including all dependencies:
    ```bash
    # create conda environment from YAML file
    conda env create -f rgbd_segmentation.yaml
    # activate environment
    conda activate rgbd_segmentation
    ```

3. Data preparation (training / evaluation / dataset inference):  
    We trained our networks on 
    [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), 
    [SUNRGB-D](https://rgbd.cs.princeton.edu/), and 
    [Cityscapes](https://www.cityscapes-dataset.com/). 
    The encoders were pretrained on [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/downloads).
    Furthermore, we also pretrained our best model on the synthetic dataset 
    [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html). 

    The folder [`src/datasets`](src/datasets) contains the code to prepare
    NYUv2, SunRGB-D, Cityscapes, SceneNet RGB-D for training and evaluation. 
    Please follow the instructions given for the respective dataset and store 
    the created datasets in `./datasets`.
    For ImageNet, we used [TensorFlowDatasets](https://www.tensorflow.org/datasets/catalog/imagenet2012) (see `imagenet_pretraining.py`).

4. Pretrained models (evaluation):  
   We provide the weights for our selected ESANet-R34-NBt1D (with ResNet34 NBt1D backbones) on NYUv2, SunRGBD, and Cityscapes:
   
   | Dataset                 | Model                            | mIoU  | FPS*     | URL  |
   |-------------------------|----------------------------------|-------|----------|------|
   | NYUv2 (test)            | ESANet-R34-NBt1D                 | 50.30 | 29.7     | [Download](https://drive.google.com/uc?id=1C5-kJv4w3foicEudP3DAjdIXVuzUK7O8) |
   |                         | ESANet-R34-NBt1D (pre. SceneNet) | 51.58 | 29.7     | [Download](https://drive.google.com/uc?id=1w_Qa8AWUC6uHzQamwu-PAqA7P00hgl8w) |
   | SUNRGB-D (test)         | ESANet-R34-NBt1D                 | 48.17 | 29.7**   | [Download](https://drive.google.com/uc?id=1tviMAEOr-6lJphpluGvdhBDA_FetIR14) |
   |                         | ESANet-R34-NBt1D (pre. SceneNet) | 48.04 | 29.7**   | [Download](https://drive.google.com/uc?id=1ukKafozmAcr8fQLbVvTtioKPLwTu0XZO) |
   | Cityscapes (valid half) | ESANet-R34-NBt1D                 | 75.22 | 23.4     | [Download](https://drive.google.com/uc?id=1xal13D_lXYVlfJx_NBiPTvuf4Ijn7wrt) |
   | Cityscapes (valid full) | ESANet-R34-NBt1D                 | 80.09 | 6.2      | [Download](https://drive.google.com/uc?id=18eKh2XD9fwdYCUM4MuCYxH7jNucIYk8O) |
   
   Download and extract the models to `./trained_models`.
   
   *We report the FPS for NVIDIA Jetson AGX Xavier (Jetpack 4.4, TensorRT 7.1, Float16).   
   **Note that we only reported the inference time for NYUv2 in our paper as it has more classes than SUNRGB-D. 
   Thus, the FPS for SUNRGB-D can be slightly higher (37 vs. 40 classes).  

### Training
Use `train.py` to train ESANet on NYUv2, SUNRGB-D, Cityscapes, or SceneNet RGB-D
(or implement your own dataset by following the implementation of the provided 
datasets).
The arguments default to training ESANet-R34-NBt1D on NYUv2 with the 
hyper-parameters from our paper. Thus, they could be omitted but are presented 
here for clarity.

> Note that training ESANet-R34-NBt1D requires the pretrained weights for the 
encoder backbone ResNet-34 NBt1D. You can download our pretrained weights on 
ImageNet from [Link](https://drive.google.com/uc?id=1neUb6SJ87dIY1VvrSGxurVBQlH8Pd_Bi). 
Otherwise, you can use `imagenet_pretraining.py` to create your own pretrained weights.

Examples: 
- Train our ESANet-R34-NBt1D on NYUv2 (except for the dataset arguments, also 
valid for SUNRGB-D):
    ```bash
    # either specify all arguments yourself
    python train.py \
        --dataset nyuv2 \
        --dataset_dir ./datasets/nyuv2 \
        --pretrained_dir ./trained_models/imagenet \
        --results_dir ./results \
        --height 480 \
        --width 640 \
        --batch_size 8 \
        --batch_size_valid 24 \
        --lr 0.01 \
        --optimizer SGD \
        --class_weighting median_frequency \
        --encoder resnet34 \
        --encoder_block NonBottleneck1D \
        --nr_decoder_blocks 3 \
        --modality rgbd \
        --encoder_decoder_fusion add \
        --context_module ppm \
        --decoder_channels_mode decreasing \
        --fuse_depth_in_rgb_encoder SE-add \
        --upsampling learned-3x3-zeropad
    
    # or use the default arguments
    python train.py \
        --dataset nyuv2 \
        --dataset_dir ./datasets/nyuv2 \
        --pretrained_dir ./trained_models/imagenet \
        --results_dir ./results
    ```

- Train our ESANet-R34-NBt1D on Cityscapes:
    ```bash
    # note that the some parameters are different
    python train.py \
        --dataset cityscapes-with-depth \
        --dataset_dir ./datasets/cityscapes \
        --pretrained_dir ./trained_models/imagenet \
        --results_dir ./results \
        --raw_depth \
        --he_init \
        --aug_scale_min 0.5 \
        --aug_scale_max 2.0 \
        --valid_full_res \
        --height 512 \
        --width 1024 \
        --batch_size 8 \
        --batch_size_valid 16 \
        --lr 1e-4 \
        --optimizer Adam \
        --class_weighting None \
        --encoder resnet34 \
        --encoder_block NonBottleneck1D \
        --nr_decoder_blocks 3 \
        --modality rgbd \
        --encoder_decoder_fusion add \
        --context_module appm-1-2-4-8 \
        --decoder_channels_mode decreasing \
        --fuse_depth_in_rgb_encoder SE-add \
        --upsampling learned-3x3-zeropad
    ```

For further information, use `python train.py --help` or take a look at 
`src/args.py`.

> To analyze the model structure, use `model_to_onnx.py` with the same 
arguments to export an ONNX model file, which can be nicely visualized using 
[Netron](https://github.com/lutzroeder/netron).

> Note that in order to facilitate parameter training, the author has added 
the function set_config_ nyuv2(), if not needed, you can choose to mask it

> Specific training process and dataset download viewing report.
