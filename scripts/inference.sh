
export PYTHONPATH=$PYTHONPATH';D:/repos/deeplab/scripts'

DATASET_FOLDER="D:/data/080420"

python ../inference.py \
    --image_folder="${DATASET_FOLDER}/coco_images" \
    --output_folder="${DATASET_FOLDER}/coco_deeplab_masks" \
    --model_directory="D:/models/deeplab_scooter_128k.tar.gz" \
    --mask_size='1280,720'


# IMPORTANT WARNING
    # when using crf, the output logits will be the edited softmax layer, NOT logits layer

# MAIN ARGUMENTS
    # --image_folder        # path to folder with images
    # --output_folder       # path to folder for segmented images
    # --model_directory     # path to the directory with tar.gz model checkpoint"
    # --mask_size           # width, height of image size (default = 513,513)
    # --output_logits       # flag to output logits instead of image data

# POST-PROCESSING
    # --crf_config          # crf pairwise potential kernal config (position, color for appearance, position for smoothing)
    # --use_crf             # flag to apply crf
    # --mark_main_road      # flag to mark the main road

# VISUALISATION ARGUMENTS
    # --vis_mask            # flag to turn mask into visualisation
    # --add_orig            # flag to attach segmentation image with original
    # --translate_labels    # flag to translate labels (e.g. sky to building)

# OTHERS
    # --print_tensor_path   # path to text file, no segmentation, only check checkpoint model
    # --use_cpu             # use CPU only