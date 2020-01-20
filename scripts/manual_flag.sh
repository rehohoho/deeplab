
export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/deeplabv3/deeplab

python /home/whizz/Desktop/deeplabv3/deeplab/post/manual_flag_pygame.py \
    --unflagged_folder="/home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-notflagged" \
    --flagged_folder="/home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-flagged" \
    --output_csv="/home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-flaggt.csv"
    
# can continue labelling by specifying --input_csv
# --input_csv="/home/whizz/Desktop/deeplabv3/pretrainseg/11-May-2019-17-54-07-flaggt.csv" \
