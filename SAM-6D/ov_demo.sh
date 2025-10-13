# Render CAD templates
cd Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 

# Run instance segmentation model
export SEGMENTOR_MODEL=fastsam

cd ../Instance_Segmentation_Model
python run_inference_custom_ov.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH


# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

FILE1="ov_pem_sub1_model.xml"
FILE2="ov_pem_sub2_model.xml"
FILE3="ov_pem_sub3_model.xml"
FILE4="ov_pem_sub4_model.xml"

cd ../Pose_Estimation_Model
mkdir output
mkdir model_save
if [ ! -f "$FILE1" ] || [ ! -f "$FILE2" ] || [ ! -f "$FILE3" ] || [ ! -f "$FILE4" ]; then
    sh pem_model_convert_gpu.sh
fi
python run_inference_custom_openvino_gpu.py --device GPU
