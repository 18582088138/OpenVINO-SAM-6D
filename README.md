# <p align="center"> <font color=#008000>SAM-6D</font>: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation </p>

####  <p align="center"> [Jiehong Lin](https://jiehonglin.github.io/), [Lihua Liu](https://github.com/foollh), [Dekun Lu](https://github.com/WuTanKun), [Kui Jia](http://kuijia.site/)</p>
#### <p align="center">CVPR 2024 </p>
#### <p align="center">[[Paper]](https://arxiv.org/abs/2311.15707) </p>

<p align="center">
  <img width="100%" src="./pics/vis.gif"/>
</p>


## News
- [2024/03/07] We publish an updated version of our paper on [ArXiv](https://arxiv.org/abs/2311.15707).
- [2024/02/29] Our paper is accepted by CVPR2024!
- [2025/08/01] OpenVINO enable SAM6D PEM model on Intel CPU [by Kunda].
- [2025/09/30] OpenVINO enable SAM6D PEM model on Intel GPU [by Kunda].
- [2025/10/05] OpenVINO enable SAM6D ISM model on Intel CPU/GPU [by Kunda].
- [2025/10/20] OpenVINO implement SAM6D E2E pipeline on Intel CPU/GPU [by Kunda].
- [2025/11/10] OpenVINO implement SAM6D [README](./README_OV.md) and [script](./SAM-6D/ov_demo.sh) Update [by Kunda].

## To Do List
- [None] 

## Update Log
- [2024/03/05] We update the demo to support [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), you can do this by specifying `SEGMENTOR_MODEL=fastsam` in demo.sh.
- [2024/03/03] We upload a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for running custom data.
- [2024/03/01] We update the released [model](https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing) of PEM. For the new model, a larger batchsize of 32 is set, while that of the old is 12. 
- [2025/08/01] Implement SAM6D onnx / openvino IR model convert, and implement the OpenVINO SAM6D PEM pipeline  [by Kunda]. 

## Overview
In this work, we employ Segment Anything Model as an advanced starting point for **zero-shot 6D object pose estimation** from RGB-D images, and propose a novel framework, named **SAM-6D**, which utilizes the following two dedicated sub-networks to realize the focused task:
- [x] [Instance Segmentation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Instance_Segmentation_Model)
- [x] [Pose Estimation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Pose_Estimation_Model)


<p align="center">
  <img width="50%" src="./pics/overview_sam_6d.png"/>
</p>


## Getting Started

**The OpenVINO implement SAM6D please reference [OpenVINO Implement SAM6D E2E pipeline](./README_OV.md)**

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/JiehongLin/SAM-6D.git
```
Install the environment and download the model checkpoints:
```
cd SAM-6D
sh prepare.sh
```
We also provide a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for convenience.

### 2. Evaluation on the custom data
```
# set the paths
export PROJECT_ROOT=$(git rev-parse --show-toplevel)
export CAD_PATH=$PROJECT_ROOT/SAM-6D/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=$PROJECT_ROOT/SAM-6D/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=$PROJECT_ROOT/SAM-6D/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$PROJECT_ROOT/SAM-6D/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$PROJECT_ROOT/SAM-6D/Data/Example/outputs         # path to a pre-defined file for saving results

# run inference
cd SAM-6D
sh demo.sh

# [Issue Fix] Imageio: 'libfreeimage-3.16.0-linux64.so' was not found on your computer; 
sudo apt-get update
sudo apt-get install libfreeimage-dev

```


Due to a PEM model struction refactor, the PEM inference script [run_inference_custom.py](./SAM-6D/Pose_Estimation_Model/run_inference_custom.py) in this branch no longer works. 

Please use run_inference_custom_pytorch.py for PyTorch CPU/CUDA inference.
            
Currently the refactor PEM model only supports model inference, not model training. 
If you need to retrain a model, pls use the original [SAM6D repository](https://github.com/JiehongLin/SAM-6D/tree/main).
This script will be removed in the future.

## OV Enable Summary
Successfully enabled the SAM6D-PEM model for CPU& GPUs on OpenVINO-2025.4.
<p align="center">
  <img width="50%" src="./pics/vis_pem_ov_GPU.png"/>
</p>

## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }


## Contact

If you have any questions, please feel free to contact the authors. 

Jiehong Lin: [mortimer.jh.lin@gmail.com](mailto:mortimer.jh.lin@gmail.com)

Lihua Liu: [lihualiu.scut@gmail.com](mailto:lihualiu.scut@gmail.com)

Dekun Lu: [derkunlu@gmail.com](mailto:derkunlu@gmail.com)

Kui Jia:  [kuijia@gmail.com](kuijia@gmail.com)

Kunda Xu: [752038@gmail.com](752038@gmail.com) [Only for OpenVINO Optimization and Enable]
