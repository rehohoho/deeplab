
DATA_FOLDER_PATH="/home/whizz/Desktop/deeplabv3/pretrainseg/seg_example"
DATASET_NAME="11-May-2019-18-29-31_mid"

python /home/whizz/Desktop/deeplabv3/deeplab/utils/video_utils.py \
    --video_dir="${DATA_FOLDER_PATH}/${DATASET_NAME}.mp4" \
    --output_folder="${DATA_FOLDER_PATH}/${DATASET_NAME}" \
    --fps=6