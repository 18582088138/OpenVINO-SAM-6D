# Init Env paths
export PROJECT_ROOT=$(git rev-parse --show-toplevel)
export CAD_PATH=$PROJECT_ROOT/SAM-6D/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=$PROJECT_ROOT/SAM-6D/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=$PROJECT_ROOT/SAM-6D/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$PROJECT_ROOT/SAM-6D/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$PROJECT_ROOT/SAM-6D/Data/Example/outputs  

export SEGMENTOR_MODEL=fastsam
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

# Render CAD templates
cd Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 

# OpenVINO Run instance segmentation model
cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH


# OV model convert
cd ../Pose_Estimation_Model
mkdir -p output
mkdir -p model_save
FILE1="model_save/ov_pem_sub1_model.xml"
FILE2="model_save/ov_pem_sub2_model.xml"
FILE3="model_save/ov_pem_sub3_model.xml"
FILE4="model_save/ov_pem_sub4_model.xml"
if [ ! -f "$FILE1" ] || [ ! -f "$FILE2" ] || [ ! -f "$FILE3" ] || [ ! -f "$FILE4" ]; then
    sh pem_model_convert_gpu.sh
fi

# OpenVINO Run pose estimation model
cd ../Pose_Estimation_Model
python run_inference_custom_openvino_gpu.py --device GPU
