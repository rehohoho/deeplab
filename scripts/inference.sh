
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab

DATASET_FOLDER="/home/whizz/Desktop/deeplabv3/pretrainseg"
DATASET_NAME="23-Apr-2019-09-09"

python /home/whizz/Desktop/deeplabv3/deeplab/inference.py \
    --image_folder="${DATASET_FOLDER}/${DATASET_NAME}" \
    --output_folder="${DATASET_FOLDER}/${DATASET_NAME}-seg_alpha" \
    --model_director="${DATASET_FOLDER}/pretrained_models" \
    --model_variant="deeplab_adambest" \
    --crf_pos=80 \
    --crf_col=26 \
    --crf_smooth=3