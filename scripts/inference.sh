
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab

DATASET_FOLDER="/home/whizz/data"

python /home/whizz/Desktop/deeplabv3/deeplab/inference.py \
    --image_folder="${DATASET_FOLDER}/transfer" \
    --output_folder="${DATASET_FOLDER}/transfer_vis_withpost" \
    --model_directory="${DATASET_FOLDER}/models/deeplab_adam_150k.tar.gz" \
    --crf_pos=80 \
    --crf_col=26 \
    --crf_smooth=3 \
    --use_crf \
    --mark_main_road \
    --mask_size="1280,960"

    
    # --print_tensor='directory/tensors.txt'                # path to text file, no segmentation, only check checkpoint model
    # --gpu=-1                                              # use CPU only
    # --use_crf                                             # flag to use crf
    # --mark_main_road                                      # flag to mark main road
    # --add_orig                                            # flag to attach segmentation image with original
    # --vis_mask                                            # flag to turn mask into visualisation
    # --translate_labels                                    # flag to translate sky to building e.g.
    # --mask_size="1280,960"                                # size of output mask, NOT APPLIED FOR VIS
    # --print_tensor="directory/checkpoint_tensors.txt"     # output tensors of chkpoint to text file