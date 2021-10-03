
export PYTHONPATH=$PYTHONPATH:C:/Users/User/Desktop/deeplabv3/deeplab

PARENT_DIR="D:/data/180820"

python D:/repos/deeplab/utils/stack_images.py \
    --root_directory="${PARENT_DIR}" \
    --folders_config='[["clipped_frames","clipped_frames_deeplab_inferred"]]' \
    --output_folder="${PARENT_DIR}/combined" \
    --font="arial.ttf"

# for windows   --font="arial.ttf"
# for linux     --font="/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"