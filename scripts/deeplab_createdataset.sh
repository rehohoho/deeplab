
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab/datasets #build_data.py

DATASET_NAME="scooter_halflabelled"
PATH_TO_DEEPLAB_DATASET="/home/whizz/Desktop/deeplabv3/deeplab/datasets"
PATH_TO_DATASET="${PATH_TO_DEEPLAB_DATASET}/${DATASET_NAME}"

#BDD PATHS, max 7000, 1000, 2000
BDD_IMAGE_PATH="/home/whizz/Desktop/deeplabv3/datasets/bdd100k/images"
BDD_SEG_PATH="/home/whizz/Desktop/deeplabv3/datasets/bdd100k/label"
BDD_IMAGE_NAME_FORMAT=".jpg"
BDD_SEG_NAME_FORMAT="_train_id.png"

#CITYSCAPES PATHS, max 2975, 500, 1525
CITYSCAPES_IMAGE_PATH="/home/whizz/Desktop/deeplabv3/datasets/cityscapes/cityscapes_images"
CITYSCAPES_SEG_PATH="/home/whizz/Desktop/deeplabv3/datasets/cityscapes/cityscapes_seg"
CITYSCAPES_IMAGE_NAME_FORMAT="_leftImg8bit.png"
CITYSCAPES_SEG_NAME_FORMAT="_gtFine_labelIds.png"

#MAPILLARY PATHS, max 18000, 2000, 5000
MAPILLARY_IMAGE_PATH="/home/whizz/Desktop/deeplabv3/datasets/mapillary/mapillary_images"
MAPILLARY_SEG_PATH="/home/whizz/Desktop/deeplabv3/datasets/mapillary/mapillary_seg"
MAPILLARY_IMAGE_NAME_FORMAT=".jpg"
MAPILLARY_SEG_NAME_FORMAT=".png"

#SCOOTER PATHS, max 4251, 906
SCOOTER_IMAGE_PATH="/home/whizz/Desktop/deeplabv3/datasets/scooter_halflabelled/scooter_images"
SCOOTER_SEG_PATH="/home/whizz/Desktop/deeplabv3/datasets/scooter_halflabelled/scooter_seg"
SCOOTER_IMAGE_NAME_FORMAT=".png"
SCOOTER_SEG_NAME_FORMAT=".png"

# python /home/whizz/Desktop/deeplabv3/deeplab/datasets/create_dataset.py \
#     --dataset_source="SCOOTER" \
#     --imagefolder_path="${SCOOTER_IMAGE_PATH}" \
#     --image_name_format="${SCOOTER_IMAGE_NAME_FORMAT}" \
#     --segfolder_path="${SCOOTER_SEG_PATH}" \
#     --seg_name_format="${SCOOTER_SEG_NAME_FORMAT}" \
#     --dataset_path="${PATH_TO_DEEPLAB_DATASET}" \
#     --dataset_name="${DATASET_NAME}" \
#     --output_format='jpg' \
#     --train=4251 \
#     --val=906 \
#     --test=0

# python /home/whizz/Desktop/deeplabv3/deeplab/datasets/create_dataset.py \
#     --dataset_source="BDD" \
#     --imagefolder_path="${BDD_IMAGE_PATH}" \
#     --image_name_format="${BDD_IMAGE_NAME_FORMAT}" \
#     --segfolder_path="${BDD_SEG_PATH}" \
#     --seg_name_format="${BDD_SEG_NAME_FORMAT}" \
#     --dataset_path="${PATH_TO_DEEPLAB_DATASET}" \
#     --dataset_name="${DATASET_NAME}" \
#     --output_format='jpg' \
#     --train=10 \
#     --val=10 \
#     --test=10

# python /home/whizz/Desktop/deeplabv3/deeplab/datasets/create_dataset.py \
#     --dataset_source="CITYSCAPES" \
#     --imagefolder_path="${CITYSCAPES_IMAGE_PATH}" \
#     --image_name_format="${CITYSCAPES_IMAGE_NAME_FORMAT}" \
#     --segfolder_path="${CITYSCAPES_SEG_PATH}" \
#     --seg_name_format="${CITYSCAPES_SEG_NAME_FORMAT}" \
#     --dataset_path="${PATH_TO_DEEPLAB_DATASET}" \
#     --dataset_name="${DATASET_NAME}" \
#     --output_format='jpg' \
#     --train=10 \
#     --val=10 \
#     --test=10

# python /home/whizz/Desktop/deeplabv3/deeplab/datasets/create_dataset.py \
#     --dataset_source="MAPILLARY" \
#     --imagefolder_path="${MAPILLARY_IMAGE_PATH}" \
#     --image_name_format="${MAPILLARY_IMAGE_NAME_FORMAT}" \
#     --segfolder_path="${MAPILLARY_SEG_PATH}" \
#     --seg_name_format="${MAPILLARY_SEG_NAME_FORMAT}" \
#     --dataset_path="${PATH_TO_DEEPLAB_DATASET}" \
#     --dataset_name="${DATASET_NAME}" \
#     --output_format='jpg' \
#     --train=10 \
#     --val=10 \
#     --test=10

# #set gpu utilise to -1 to disable
python /home/whizz/Desktop/deeplabv3/deeplab/datasets/build_dataset.py \
    --image_folder="${PATH_TO_DATASET}/images" \
    --image_format="jpg" \
    --semantic_segmentation_folder="${PATH_TO_DATASET}/seg" \
    --label_format='_label.png' \
    --list_folder="${PATH_TO_DATASET}/index" \
    --output_dir="${PATH_TO_DATASET}/tfrecords" \
    --gpu_utilise=0