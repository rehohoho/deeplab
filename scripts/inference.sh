
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab

DATASET_FOLDER="/home/whizz/Desktop/deeplabv3/pretrainseg"
DATASET_NAME="11-May-2019-18-29-31_mid"

python /home/whizz/Desktop/deeplabv3/deeplab/inference.py \
    --image_folder="${DATASET_FOLDER}/seg_example/${DATASET_NAME}" \
    --output_folder="${DATASET_FOLDER}/seg_example/${DATASET_NAME}-seg" \
    --model_director="${DATASET_FOLDER}/pretrained_models" \
    --model_variant="deeplab_adambest" \
    --crf_pos=80 \
    --crf_col=26 \
    --crf_smooth=3 \
    --post=1 \
    --comparison=0 \
    --gpu="0" \
    --print_tensor=""

# print_tensor: path to text file, no segmentation, only check checkpoint model
# post:         boolean if any CRF post is wanted
# gpu:          set to "-1" if want to use CPU only