'''
Quick and dirty hardcoded stacking script
1) Stacks images according to array of paths
2) Output images into a folder

Current requirements:
    - folders are not nested
    - image names are the same throughout
'''

import os
import numpy as np
import re
import json
from PIL import Image, ImageFont, ImageDraw
import argparse


def numericalSort(filenames):
    ''' sort filenames in directory by index, then position
    ensure temporal continuity for post methods that use previous frames
    '''

    new_filenames = []
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            ind = re.findall(r'\d+', filename)
            pos = ''
            if 'left' in filename:
                pos = 'left'
            if 'mid' in filename:
                pos = 'mid'
            if 'right' in filename:
                pos = 'right'
            new_filenames.append([filename, int(ind[-1]), pos])
    
    new_filenames.sort(key=lambda x:x[1])
    new_filenames.sort(key=lambda x:x[2])
    new_filenames = [i[0] for i in new_filenames]
    return(new_filenames)


def draw_text_on_combined_image(image, path_arr, font, height, width):
    
    draw = ImageDraw.Draw(image)
    y = 0

    for folder_row in path_arr:
        
        x = 0
        
        for folder_path in folder_row:
            folder_base_path = os.path.basename(folder_path)
            draw.text((x + 10, y + 10), folder_base_path, (0,0,255), font=font)
            x += width

        y += height


def stack_folders_of_images(path_arr, output_folder, font, get_right_side=False):
    ''' stack images according to filename in non-nested folders 
    
    Args:
        path_arr        2D list     contains paths to folders containing images
        output_folder   path        folder to save images to
        get_right_side  bool        flag to crop image, takes left side for first image (orig)    
    '''
    
    font = ImageFont.truetype(font, 36, encoding="unic")

    ref_folder = path_arr[0][0]
    filenames = os.listdir(ref_folder)
    filenames = numericalSort(filenames)

    ref_image_path = os.path.join(ref_folder, filenames[0] )
    width, height = Image.open(ref_image_path).size
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        print('Directory created at %s' %output_folder)

    for filename in filenames:
        
        newim = []
        orig = True

        for folder_row in path_arr:
            
            imrow = []

            for folder_path in folder_row:
                
                img_path = os.path.join( folder_path, filename )
                img = Image.open(img_path)

                if get_right_side:

                    if orig:
                        img = img.crop((0, 0, width/2, height))
                        orig = False
                    else:
                        img = img.crop((width/2, 0, width, height))
                
                imrow.append( np.array(img) )

            imrow = np.hstack(imrow)
            newim.append(imrow)
        
        newim = np.vstack(newim)
        save_path = os.path.join(output_folder, filename)
        newim = Image.fromarray(newim)
        
        if get_right_side:
            single_img_width = width/2
        else:
            single_img_width = width

        draw_text_on_combined_image(newim, path_arr, font, height, single_img_width)
        newim.save(save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_directory',
        required=True,
        help= 'directory to folders'
    )
    parser.add_argument(
        '--folders_config',
        required=True,
        help= '2D array of folder names'
    )
    parser.add_argument(
        '--output_folder',
        required=True,
        help= 'path to folder to output images'
    )
    parser.add_argument(
        '--font',
        required=True,
        help= 'path to font'
    )
    parser.add_argument(
        '--get_right_side',
        default=False,
        action='store_true',
        help= 'flag to crop to right side image'
    )

    args = parser.parse_args()

    print('Note: require folders to be not nested (depth=1)\n \
                 and image base names to be same through folders.')

    args.folders_config = json.loads(args.folders_config)
    
    path_arr = []
    for row in args.folders_config:
        path_arr.append([
            os.path.join(args.root_directory, folder_name) for folder_name in row
        ])    

    stack_folders_of_images(path_arr, args.output_folder, args.font, args.get_right_side)