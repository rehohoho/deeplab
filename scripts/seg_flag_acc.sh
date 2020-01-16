
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab

#Shell script to open up a pygame console to visualise recorded data or predicted data
python post/seg_flag_acc.py \
-i /home/whizz/data/seg_test_results \
-f1 /home/whizz/data/seg_test_post/probably_crap.csv \
-f2 /home/whizz/data/manual_check.txt