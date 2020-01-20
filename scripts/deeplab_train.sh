
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/slim

#model variant: see arg_scopes_map in core/feature_extractor.py for more backbones
#reproduce cityscapes results: batchsize>8, fine_tune_batch_norm=True
#rates: outputstride 16 (6,12,18), outputstride 8 (12,24,36)
#decoder_output_stride: use decoder structure
#dense prediction cell:
    #--model_variant="xception_71"
    #--dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json"
    #--decoder_output_stride=4
#fine tune batch norm consumes tons of gpu memory and has to use outputstride > 8
#for train batch size 8, learning rate = 0.001

# eval_size = output_stride * k + 1 (check max height and max width
# sudo nvidia-smi -i 0 -pl 210

PATH_TO_INITIAL_CHECKPOINT=/home/whizz/Desktop/deeplabv3/pretrainseg/pretrained_models/xception71/model.ckpt
DATASET_NAME="bdd+cityscapes+mapillary"
PATH_TO_DATASET="/home/whizz/Desktop/deeplabv3/deeplab/datasets/${DATASET_NAME}"
PATH_TO_TFRECORDS="${PATH_TO_DATASET}/tfrecords"

#mom lr : best 2e-4 from MIOU, second best 1e-4
#adam lr: best 1e-5 from mIoU, second best 5e-6 (better than mom)



# control
# LOOPS=( 0 1 )
LOOPS=( 0 )
for i in "${LOOPS[@]}"
do
    num=$(expr $i | bc)
    python /home/whizz/Desktop/deeplabv3/deeplab/train.py \
        --logtostderr \
        --training_number_of_steps=150000 \
        --train_split="train" \
        --model_variant="xception_71" \
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --train_crop_size="513,513" \
        --train_batch_size=2 \
        --fine_tune_batch_norm=False \
        --dataset=${DATASET_NAME} \
        --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
        --train_logdir="${PATH_TO_DATASET}/mom_${num}_longrun" \
        --dataset_dir=${PATH_TO_TFRECORDS} \
        --min_scale_factor=0.5 \
        --max_scale_factor=2 \
        --motion_blur_size=0 \
        --motion_blur_direction_limit=30 \
        --rotation_min_limit=0 \
        --rotation_max_limit=0 \
        --brightness_min_limit=0 \
        --brightness_max_limit=0 \
        --save_interval_secs=800 \
        --initialize_last_layer=True \
        --last_layers_contain_logits_only=False \
        --save_summaries_images=True \
        --base_learning_rate=0.000001

    python /home/whizz/Desktop/deeplabv3/deeplab/run_during_eval.py \
    --checkpoint_dir="${PATH_TO_DATASET}/mom_${num}_longrun" \
    --event_dir="${PATH_TO_DATASET}/mom_${num}_longrun_eval" &

    python /home/whizz/Desktop/deeplabv3/deeplab/eval.py \
        --logtostderr \
        --eval_split="val" \
        --model_variant="xception_71" \
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --eval_crop_size="1025,2049" \
        --dataset=${DATASET_NAME} \
        --checkpoint_dir="${PATH_TO_DATASET}/mom_${num}_longrun" \
        --eval_logdir="${PATH_TO_DATASET}/mom_${num}_longrun_eval" \
        --dataset_dir=${PATH_TO_TFRECORDS}
done



# rotation
python /home/whizz/Desktop/deeplabv3/deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=150000 \
    --train_split="train" \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=2 \
    --fine_tune_batch_norm=False \
    --dataset=${DATASET_NAME} \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir="${PATH_TO_DATASET}/mom_1_rot" \
    --dataset_dir=${PATH_TO_TFRECORDS} \
    --min_scale_factor=0.5 \
    --max_scale_factor=2 \
    --motion_blur_size=0 \
    --motion_blur_direction_limit=30 \
    --rotation_min_limit=-30 \
    --rotation_max_limit=30 \
    --brightness_min_limit=0 \
    --brightness_max_limit=0 \
    --save_interval_secs=800 \
    --initialize_last_layer=True \
    --last_layers_contain_logits_only=False \
    --save_summaries_images=True \
    --base_learning_rate=0.000001

python /home/whizz/Desktop/deeplabv3/deeplab/run_during_eval.py \
--checkpoint_dir="${PATH_TO_DATASET}/mom_1_rot" \
--event_dir="${PATH_TO_DATASET}/mom_1_rot_eval" &

python /home/whizz/Desktop/deeplabv3/deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --dataset=${DATASET_NAME} \
    --checkpoint_dir="${PATH_TO_DATASET}/mom_1_rot" \
    --eval_logdir="${PATH_TO_DATASET}/mom_1_rot_eval" \
    --dataset_dir=${PATH_TO_TFRECORDS}



# blurring
python /home/whizz/Desktop/deeplabv3/deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=150000 \
    --train_split="train" \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=2 \
    --fine_tune_batch_norm=False \
    --dataset=${DATASET_NAME} \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir="${PATH_TO_DATASET}/mom_1_blur" \
    --dataset_dir=${PATH_TO_TFRECORDS} \
    --min_scale_factor=0.5 \
    --max_scale_factor=2 \
    --motion_blur_size=0.05 \
    --motion_blur_direction_limit=30 \
    --rotation_min_limit=0 \
    --rotation_max_limit=0 \
    --brightness_min_limit=0 \
    --brightness_max_limit=0 \
    --save_interval_secs=800 \
    --initialize_last_layer=True \
    --last_layers_contain_logits_only=False \
    --save_summaries_images=True \
    --base_learning_rate=0.000001

python /home/whizz/Desktop/deeplabv3/deeplab/run_during_eval.py \
--checkpoint_dir="${PATH_TO_DATASET}/mom_1_blur" \
--event_dir="${PATH_TO_DATASET}/mom_1_blur_eval" &

python /home/whizz/Desktop/deeplabv3/deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --dataset=${DATASET_NAME} \
    --checkpoint_dir="${PATH_TO_DATASET}/mom_1_blur" \
    --eval_logdir="${PATH_TO_DATASET}/mom_1_blur_eval" \
    --dataset_dir=${PATH_TO_TFRECORDS}



# brightness
python /home/whizz/Desktop/deeplabv3/deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=150000 \
    --train_split="train" \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=2 \
    --fine_tune_batch_norm=False \
    --dataset=${DATASET_NAME} \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir="${PATH_TO_DATASET}/mom_1_bright" \
    --dataset_dir=${PATH_TO_TFRECORDS} \
    --min_scale_factor=0.5 \
    --max_scale_factor=2 \
    --motion_blur_size=0 \
    --motion_blur_direction_limit=30 \
    --rotation_min_limit=0 \
    --rotation_max_limit=0 \
    --brightness_min_limit=0.5 \
    --brightness_max_limit=1.5 \
    --save_interval_secs=800 \
    --initialize_last_layer=True \
    --last_layers_contain_logits_only=False \
    --save_summaries_images=True \
    --base_learning_rate=0.000001

python /home/whizz/Desktop/deeplabv3/deeplab/run_during_eval.py \
--checkpoint_dir="${PATH_TO_DATASET}/mom_1_bright" \
--event_dir="${PATH_TO_DATASET}/mom_1_bright_eval" &

python /home/whizz/Desktop/deeplabv3/deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="1025,2049" \
    --dataset=${DATASET_NAME} \
    --checkpoint_dir="${PATH_TO_DATASET}/mom_1_bright" \
    --eval_logdir="${PATH_TO_DATASET}/mom_1_bright_eval" \
    --dataset_dir=${PATH_TO_TFRECORDS}














# LEARNING RATE TUNING
#1e-7 to 1e-5, default 1e-3 epsilon 1e-8
#best 1e-5 from mIoU, second best 5e-6
# ADAM=( 1 3 10 30 100 300 1000 )
# adamMult=0.00001  #1e-5 to 1e-2

# for i in "${ADAM[@]}"
# do
#     num=$(expr $adamMult*$i | bc)
#     python /home/whizz/Desktop/deeplabv3/deeplab/train.py \
#         --logtostderr \
#         --training_number_of_steps=5000 \
#         --train_split="train" \
#         --model_variant="xception_71" \
#         --atrous_rates=6 \
#         --atrous_rates=12 \
#         --atrous_rates=18 \
#         --output_stride=16 \
#         --decoder_output_stride=4 \
#         --train_crop_size="513,513" \
#         --train_batch_size=2 \
#         --fine_tune_batch_norm=False \
#         --dataset=${DATASET_NAME} \
#         --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
#         --train_logdir="${PATH_TO_DATASET}/adam_${num}" \
#         --dataset_dir=${PATH_TO_TFRECORDS} \
#         --min_scale_factor=0.5 \
#         --max_scale_factor=2 \
#         --motion_blur_size=0 \
#         --motion_blur_direction_limit=30 \
#         --rotation_min_limit=0 \
#         --rotation_max_limit=0 \
#         --brightness_min_limit=0 \
#         --brightness_max_limit=0 \
#         --save_interval_secs=400 \
#         --initialize_last_layer=True \
#         --last_layers_contain_logits_only=False \
#         --save_summaries_images=True \
#         --optimizer='adam' \
#         --adam_epsilon=1e-03 \
#         --adam_learning_rate=$num
    
#     python /home/whizz/Desktop/deeplabv3/deeplab/run_during_eval.py \
#     --checkpoint_dir="${PATH_TO_DATASET}/adam_${num}" \
#     --event_dir="${PATH_TO_DATASET}/adam_${num}_eval" &
    
#     python /home/whizz/Desktop/deeplabv3/deeplab/eval.py \
#         --logtostderr \
#         --eval_split="val" \
#         --model_variant="xception_71" \
#         --atrous_rates=6 \
#         --atrous_rates=12 \
#         --atrous_rates=18 \
#         --output_stride=16 \
#         --decoder_output_stride=4 \
#         --eval_crop_size="1025,2049" \
#         --dataset=${DATASET_NAME} \
#         --checkpoint_dir="${PATH_TO_DATASET}/adam_${num}" \
#         --eval_logdir="${PATH_TO_DATASET}/adam_${num}_eval" \
#         --dataset_dir=${PATH_TO_TFRECORDS}
# done

#1e-6 to 1e-3, default 1e-4
#best 2e-4 from MIOU, second best 1e-4
# MOM=( 2 5 10 20 50 100 200 500 1000 )
# momMult=0.000001

# for i in "${MOM[@]}"
# do
#     num=$(expr $momMult*$i | bc)
#     python /home/whizz/Desktop/deeplabv3/deeplab/train.py \
#         --logtostderr \
#         --training_number_of_steps=5000 \
#         --train_split="train" \
#         --model_variant="xception_71" \
#         --atrous_rates=6 \
#         --atrous_rates=12 \
#         --atrous_rates=18 \
#         --output_stride=16 \
#         --decoder_output_stride=4 \
#         --train_crop_size="513,513" \
#         --train_batch_size=2 \
#         --fine_tune_batch_norm=False \
#         --dataset=${DATASET_NAME} \
#         --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
#         --train_logdir="${PATH_TO_DATASET}/mom_${num}" \
#         --dataset_dir=${PATH_TO_TFRECORDS} \
#         --min_scale_factor=0.5 \
#         --max_scale_factor=2 \
#         --motion_blur_size=0 \
#         --motion_blur_direction_limit=30 \
#         --rotation_min_limit=0 \
#         --rotation_max_limit=0 \
#         --brightness_min_limit=0 \
#         --brightness_max_limit=0 \
#         --save_interval_secs=400 \
#         --initialize_last_layer=True \
#         --last_layers_contain_logits_only=False \
#         --save_summaries_images=True \
#         --base_learning_rate=$num
    
#     python /home/whizz/Desktop/deeplabv3/deeplab/run_during_eval.py \
#     --checkpoint_dir="${PATH_TO_DATASET}/mom_${num}" \
#     --event_dir="${PATH_TO_DATASET}/mom_${num}_eval" &
    
#     python /home/whizz/Desktop/deeplabv3/deeplab/eval.py \
#         --logtostderr \
#         --eval_split="val" \
#         --model_variant="xception_71" \
#         --atrous_rates=6 \
#         --atrous_rates=12 \
#         --atrous_rates=18 \
#         --output_stride=16 \
#         --decoder_output_stride=4 \
#         --eval_crop_size="1025,2049" \
#         --dataset=${DATASET_NAME} \
#         --checkpoint_dir="${PATH_TO_DATASET}/mom_${num}" \
#         --eval_logdir="${PATH_TO_DATASET}/mom_${num}_eval" \
#         --dataset_dir=${PATH_TO_TFRECORDS}
# done