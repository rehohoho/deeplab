
export PYTHONPATH=$PYTHONPATH:C:/Users/User/Desktop/deeplabv3/deeplab

PARENT_DIR="C:/Users/User/Desktop/data/29-Apr-2019-12-05"

python C:/Users/User/Desktop/deeplab/utils/stack_images.py \
    --root_directory="${PARENT_DIR}" \
    --folders_config='[["transfer_scooteronly_nopost_noresize", "transfer_scooteronly_withpost_noresize"],["fixed_post_nopost_noresize", "fixed_post_withpost_noresize"]]' \
    --output_folder="${PARENT_DIR}/combined" \
    --font="arial.ttf" \
    --get_right_side

# for windows   --font="arial.ttf"
# for linux     --font="/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"