"""
Quick and dirty hardcoded stacking script
"""


import os
import numpy as np
import re
from PIL import Image, ImageFont, ImageDraw

def numericalSort(filenames):
    """ sort filenames in directory by index, then position
    ensure temporal continuity for post methods that use previous frames
    """

    new_filenames = []
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            ind = re.findall(r'\d+', filename)
            pos = None
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


folders = ['23-Apr-2019-09-09_seg',
            '23-Apr-2019-09-09_mom5seg', '23-Apr-2019-09-09_mom5segpost']

folders = ['11-May-2019-17-54-07-seg_col_13', '11-May-2019-17-54-07-seg_col_26', '11-May-2019-17-54-07-seg_col_52', 
        '11-May-2019-17-54-07-seg_pos_40', '11-May-2019-17-54-07-seg_pos_80', '11-May-2019-17-54-07-seg_pos_160', 
        '11-May-2019-17-54-07-seg_smooth_1', '11-May-2019-17-54-07-seg_smooth_3', '11-May-2019-17-54-07-seg_smooth_6'] 

orig = '11-May-2019-17-54-07-seg_alph'

f = open('crf_metrics_0.3minprob.csv', 'r')
stats = f.read()
f.close()
stats = stats.split('\n')

filenames = os.listdir(folders[0])
filenames = numericalSort(filenames)
i = 0
height, width = np.shape( np.array(Image.open(os.path.join(folders[0], filenames[0] ))))[:2]
font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 36, encoding="unic")

for filename in filenames:
    newim = []
    
    row = np.hstack((
        np.array(Image.open( os.path.join(folders[0], filename )).crop((0, 0, width/2, height))),
        np.array(Image.open( os.path.join(folders[0], filename )).crop((width/2, 0, width, height))),
    ))
    newim.append(row)
    row = np.hstack((
        np.array(Image.open( os.path.join(folders[1], filename )).crop((width/2, 0, width, height))),
        np.array(Image.open( os.path.join(folders[2], filename )).crop((width/2, 0, width, height)))
    ))
    newim.append(row)
    newim = np.vstack( (newim[0], newim[1]) )
    newim = Image.fromarray(newim)
    newim.save( os.path.join('combined', filename) )
    i += 1

for filename in filenames:
    newim = []
    
    row = np.hstack((
        np.array(Image.open( os.path.join(folders[0], filename )).crop((0, 0, width/2, height))),
        np.array(Image.open( os.path.join(folders[0], filename )).crop((width/2, 0, width, height))),
        np.array(Image.open( os.path.join(folders[1], filename )).crop((width/2, 0, width, height))),
        np.array(Image.open( os.path.join(folders[2], filename )).crop((width/2, 0, width, height)))
    ))
    newim.append(row)

    row = np.hstack((
        np.array(Image.open( os.path.join(orig, filename )).crop((width/2, 0, width, height)).convert('RGB')),
        np.array(Image.open( os.path.join(folders[3], filename )).crop((width/2, 0, width, height))),
        np.array(Image.open( os.path.join(folders[4], filename )).crop((width/2, 0, width, height))),
        np.array(Image.open( os.path.join(folders[5], filename )).crop((width/2, 0, width, height)))
    ))
    newim.append(row)

    row = np.hstack((
        np.array(Image.open( os.path.join(folders[6], filename )).crop((0, 0, width/2, height))),
        np.array(Image.open( os.path.join(folders[6], filename )).crop((width/2, 0, width, height))),
        np.array(Image.open( os.path.join(folders[7], filename )).crop((width/2, 0, width, height))),
        np.array(Image.open( os.path.join(folders[8], filename )).crop((width/2, 0, width, height)))
    ))
    newim.append(row)

    newim = np.vstack( (newim[0], newim[1], newim[2]) )

    newim = Image.fromarray(newim)
    draw = ImageDraw.Draw(newim)
    draw.text((0,0), stats[i].replace(',', '\n'), (255,0,0), font=font)
    newim.save( os.path.join('combined', filename) )
    
    i += 1