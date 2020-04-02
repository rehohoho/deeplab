
export PYTHONPATH=$PYTHONPATH:C:/Users/User/Desktop/deeplab

DATASET_FOLDER="D:/perception_datasets/scooter_halflabelled/"

python ../inference.py \
    --image_folder="${DATASET_FOLDER}/scooter_images" \
    --output_folder="${DATASET_FOLDER}/scooter_softmax" \
    --model_directory="C:/Users/User/Downloads/deploy_models/deeplab_adam_150k.tar.gz" \
    --softmax_temp=10.0 \
    --output_logits \
    --mask_size='1280,960' \
    --crf_config='80,26,3' \
    --use_crf


# MAIN ARGUMENTS
    # --image_folder        # path to folder with images
    # --output_folder       # path to folder for segmented images
    # --model_directory     # path to the directory with tar.gz model checkpoint"
    # --mask_size           # width, height of image size (default = 513,513)
    # --softmax_temp        # temperature of softmax
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