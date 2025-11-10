### Create conda environment
# conda env create -f ov_environment.yaml
# conda activate ov_sam6d
### IF conda env create failed, we can use manual create conda env and install pip requirements.txt to setup the venv
### conda create -n ov_sam6d python=3.11
### conda activate ov_sam6d
pip install -r requirements.txt

# Due to, the special version OpenVINO we need, the default ov should uninstall
pip uninstall -y openvino openvino-tokenizers openvino-telemetry 

### Install pointnet2
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
cd ../../../

### Download ISM pretrained model
cd Instance_Segmentation_Model
python download_sam.py
python download_fastsam.py
python download_dinov2.py
cd ../

## Modify ISM submodel struction
cd Instance_Segmentation_Model/model/
# MODEL_PATH="<MODAL_SAVE_DIR>/fastsam_yolo_v8_predictor.xml"
MODEL_PATH="/home/benchmark/wayne/xkd/frameworks.industrial.motion-control.sam_6d_openvino/SAM-6D/Instance_Segmentation_Model/fastsam_yolo_v8_predictor.xml"
# Such as : MODEL_PATH="/home/benchmark/wayne/xkd/frameworks.industrial.motion-control.sam_6d_openvino/SAM-6D/Instance_Segmentation_Model/fastsam_yolo_v8_predictor.xml"
sed "s|fastsam_yolo_v8_predictor_model_path = \"\"|fastsam_yolo_v8_predictor_model_path = \"$MODEL_PATH\"|" ov_predictor.py > tmp.py
ULTRALYTICS_PATH=$(python -c "import ultralytics, os; print(os.path.dirname(ultralytics.__file__))")
PREDICTOR_FILE_PATH="$ULTRALYTICS_PATH/yolo/engine/predictor.py"
cp tmp.py "$PREDICTOR_FILE_PATH"
echo "[Logging] ultralytics predictor.py replace success $PREDICTOR_FILE_PATH"
cd ../../

## Download PEM pretrained model
cd Pose_Estimation_Model
python download_sam6d-pem.py
cd ../

### [!!!] Make Sure OpenVINO Environment Setting : source <OV_DIR>/setupvars.sh 
### Compile OV Custom Op
cd Pose_Estimation_Model/model/ov_pointnet2_op
mkdir -p build
cd build
cmake .. 
make -j8
cd ../../../
