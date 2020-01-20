
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/slim

DATASET_NAME="bdd"
PATH_TO_DATASET="/home/whizz/Desktop/deeplabv3/deeplab/datasets/${DATASET_NAME}"
PATH_TO_TFRECORDS="${PATH_TO_DATASET}/tfrecords"

python /home/whizz/Desktop/deeplabv3/deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size="1025,2049" \
    --dataset="bdd" \
    --colormap_type="cityscapes" \
    --checkpoint_dir="${PATH_TO_DATASET}/log1" \
    --vis_logdir="${PATH_TO_DATASET}/log1_vis" \
    --dataset_dir=${PATH_TO_TFRECORDS} \
    --also_save_raw_predictions=true