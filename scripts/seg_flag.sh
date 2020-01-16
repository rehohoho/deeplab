
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab

python post/seg_flag.py \
-i /home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-seg \
-o /home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-notflagged \
-t /home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-flagged \
#-f1 0.6 \ #threshold for wrong road (arbitrary ratio ~1)   (check quality of raw segmentation)
#-f2 0.5 \ #min threshold for road boundary size wrt height (check if road is large enough)
#-f3 0.9   #max threshold for road width wrt image width    (check if road has a direction)