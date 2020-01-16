
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab

#Shell script to open up a pygame console to visualise recorded data or predicted data
python post/manual_flag_pygame.py \
-t /home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-notflagged \
-f /home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-flagged \
-w /home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-flaggt.csv
