
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/slim

OUTPUT_DIR="/home/whizz/data/models"
CHECKPOINT_NAME="model.ckpt-150000"
OUTPUT_MODEL_NAME="deeplab_adam_150k"

mkdir "${OUTPUT_DIR}/temp"

python /home/whizz/Desktop/deeplabv3/deeplab/export_model.py \
    --checkpoint_path="${OUTPUT_DIR}/${CHECKPOINT_NAME}" \
    --export_path="${OUTPUT_DIR}/temp/frozen_inference_graph.pb" \
    --num_classes=19 \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4

tar -zcvf "${OUTPUT_DIR}/${OUTPUT_MODEL_NAME}.tar.gz" "${OUTPUT_DIR}/temp"

rm -R "${OUTPUT_DIR}/temp"