
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab

python /home/whizz/Desktop/deeplabv3/deeplab/post/seg_flag.py \
    --input_folder="/home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-seg" \
    --unflagged_folder="/home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-notflagged2" \
    --flagged_folder="/home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-flagged2" \
    --road_wrongness=0.6 \
    --road_minimum_size=0.5 \
    --road_maximum_width=0.9 

#threshold for wrong road (arbitrary ratio ~1)   (check quality of raw segmentation)
#min threshold for road boundary size wrt height (check if road is large enough)
#max threshold for road width wrt image width    (check if road has a direction)