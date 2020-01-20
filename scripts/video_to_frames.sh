
DATA_FOLDER_PATH="/home/whizz/Desktop/deeplabv3/pretrainseg"
DATASET_NAME="23-Apr-2019-09-09"

python /home/whizz/Desktop/deeplabv3/deeplab/utils/video_utils.py \
    --frames_folder="${DATA_FOLDER_PATH}/${DATASET_NAME}_combined" \
    --video_path="${DATA_FOLDER_PATH}/${DATASET_NAME}2.mp4"