

export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/slim

DATASET_NAME="bdd+cityscapes+mapillary"
PATH_TO_DATASET="/home/whizz/Desktop/deeplabv3/deeplab/datasets/${DATASET_NAME}"
PATH_TO_TFRECORDS="${PATH_TO_DATASET}/tfrecords"

python run_during_eval.py \
    --checkpoint_dir="${PATH_TO_DATASET}/adam_.0000001" \
    --event_dir="${PATH_TO_DATASET}/adam_.0000001_eval" &
    
python eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --dataset=${DATASET_NAME} \
    --checkpoint_dir="${PATH_TO_DATASET}/adam_.0000001" \
    --eval_logdir="${PATH_TO_DATASET}/adam_.0000001_eval" \
    --dataset_dir=${PATH_TO_TFRECORDS}

