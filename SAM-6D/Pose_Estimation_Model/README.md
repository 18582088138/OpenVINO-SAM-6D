# Pose Estimation Model (PEM) for SAM-6D 



![image](https://github.com/JiehongLin/SAM-6D/blob/main/pics/overview_pem.png)

## Requirements
The code has been tested with
- python 3.9.6
- pytorch 2.0.0
- CUDA 11.3

Other dependencies:

```
sh dependencies.sh
```

## Data Preparation

Please refer to [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data)] for more details.


## Model Download
Our trained model is provided [[here](https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing)], and could be downloaded via the command:
```
python download_sam6d-pem.py
```

## Training on MegaPose Training Set

To train the Pose Estimation Model of SAM-6D, please prepare the training data and run the folowing command:
```
python train.py --gpus 0,1,2,3 --model pose_estimation_model --config config/base.yaml
```
By default, we use four GPUs of 3090ti to train the model with batchsize set as 28.


## Evaluation on BOP Datasets

To evaluate the model on BOP datasets, please run the following command:
```
python test_bop.py --gpus 0 --model pose_estimation_model --config config/base.yaml --dataset $DATASET --view 42
```
The string "DATASET" could be set as `lmo`, `icbin`, `itodd`, `hb`, `tless`, `tudl`, `ycbv`, or `all`. Before evaluation, please refer to [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data)] for rendering the object templates of BOP datasets, or download our [rendered templates](https://drive.google.com/drive/folders/1fXt5Z6YDPZTJICZcywBUhu5rWnPvYAPI?usp=drive_link). Besides, the instance segmentation should be done following [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Instance_Segmentation_Model)]; to test on your own segmentation results, you could change the "detection_paths" in the `test_bop.py` file.

One could also download our trained model for evaluation:
```
python test_bop.py --gpus 0 --model pose_estimation_model --config config/base.yaml --checkpoint_path checkpoints/sam-6d-pem-base.pth --dataset $DATASET --view 42
```


## (Optinal) OpenVINO enable SAM6D PEM on Intel CPU
Download OpenVINO packages from [OpenVINO Archives](https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.2/linux)

### step 1. OpenVINO install 
```
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.2/linux/openvino_toolkit_ubuntu22_2025.2.0.19140.c01cd93e24d_x86_64.tgz
tar -zxvf openvino_toolkit_ubuntu22_2025.2.0.19140.c01cd93e24d_x86_64.tgz

# Setup ov environment variables
source openvino_toolkit_ubuntu22_2025.2.0.19140.c01cd93e24d_x86_64/setupvars.sh
```

### step 2. OpenVINO PEM model convert for CPU
OpenVINO custom op need to be compiled by source code, make sure the ov environment variables has already setup.
```
cd <SAM6D_DIR>/SAM-6D/Pose_Estimation_Model/model/ov_pointnet2_op/

mkdir build && cd build

cmake .. && make -j

cd <SAM6D_DIR>/SAM-6D/Pose_Estimation_Model

python pem_model_convert_cpu.py

```

### step 3. OpenVINO PEM model inference on CPU
```
python run_inference_custom_openvino_cpu.py

# to compare result with pytorch model

python run_inference_custom_pytorch.py 
```
Due to a PEM model struction refactor, the PEM inference script [run_inference_custom.py](./SAM-6D/Pose_Estimation_Model/run_inference_custom.py) in this branch no longer works. 

Please use run_inference_custom_pytorch.py for PyTorch CPU/CUDA inference.
            
Currently the refactor PEM model only supports model inference, not model training. 
If you need to retrain a model, pls use the original [SAM6D repository](https://github.com/JiehongLin/SAM-6D/tree/main).
This script will be removed in the future.


## (Optinal) OpenVINO enable SAM6D PEM on Intel GPU
The specific version of OpenVINO is required to support the SAM6D on Intel GPUs.
You must manually compile the OpenVINO source code and install it.
The following steps explain how to compile the source code and run the sam6d-pem model using OpenVINO GPUs.
- OV spec repo : https://github.com/18582088138/xkd-openvino/tree/ov_sam6d_mix

### step 1. OpenVINO spec source code compile

OpenVINO source code build for linux reference doc :  https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md
```
git clone -b ov_sam6d_mix https://github.com/18582088138/xkd-openvino
git submodule update --init --recursive

cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DCMAKE_INSTALL_PREFIX=../ov_dist_mix  -DENABLE_WHEEL=ON -DENABLE_SYSTEM_TBB=OFF  -DENABLE_DEBUG_CAPS=ON -DENABLE_GPU_DEBUG_CAPS=ON  ..

make -j8
make install

# Setup ov environment variables
source ../ov_dist_mix/setupvars.sh

```

### step 2. OpenVINO PEM model convert for GPU
OpenVINO custom op need to be compiled by source code, make sure the ov environment variables has already setup.
```
cd <SAM6D_DIR>/SAM-6D/Pose_Estimation_Model/model/ov_pointnet2_op/

mkdir build && cd build

cmake .. && make -j

cd <SAM6D_DIR>/SAM-6D/Pose_Estimation_Model

chmod 777 pem_model_convert_gpu.sh
./pem_model_convert_gpu.sh

```

### step 3. OpenVINO PEM model inference on GPU
```
python run_inference_custom_openvino_gpu.py

# to compare result with pytorch model

python run_inference_custom_pytorch.py 
```
## OV Enable Summary
Successfully enabled the SAM6D-PEM model for CPU& GPUs on OpenVINO-2025.4.
<p align="center">
  <img width="50%" src="../../pics/vis_pem_ov_GPU.png"/>
</p>

## Acknowledgements
- [MegaPose](https://github.com/megapose6d/megapose6d)
- [GDRNPP](https://github.com/shanice-l/gdrnpp_bop2022)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [Flatten Transformer](https://github.com/LeapLabTHU/FLatten-Transformer)

