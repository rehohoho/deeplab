import cv2
from cv2 import imwrite, COLOR_RGB2BGR, cvtColor
import os

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from math import ceil
import argparse
import re

import sys
#FFMPEG_DIR = 'C:\\Program Files\\FFmpeg\\bin'
#sys.path.append(FFMPEG_DIR)

import subprocess

def getFrameFromVideo(viddir, outputdir, fps):
    """ Extracts frames from video
    dir format has to use \\
    """
    
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
	
    vidcap = cv2.VideoCapture(viddir)
    success, image = vidcap.read()
    count = 0
    while success:

        if count % fps == 0:
            temp = cv2.imwrite( os.path.join(outputdir, '%s.png' %count), image)
        success, image = vidcap.read()
        count += 1

    print('Total of %s frames.' %count)
    return(count)


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


def rename_images_in_folder(folder):
    """ Rename all image names to increasing digits for ffmpeg """

    # checks if file names are already digits
    if os.path.basename(os.listdir(folder)[0]).split('.')[0].isdigit():
        return

    filenames = os.listdir(folder)
    filenames = numericalSort(filenames)

    i = 0
    for filename in filenames:
        
        oldname = os.path.join( folder, os.path.basename(filename) )
        newname = os.path.join( folder, '%s.png' %i)
        os.rename( oldname, newname )
        
        i += 1


def getVideoFromFrame(imgdir, outputdir, filetype):
    """ Forms video from jpg frames

    dir format has to use '\\'
    dependencies: ffmpeg, add to path
    """
    
    rename_images_in_folder(imgdir)

    osImgDir = os.path.join(imgdir, '%d.{}'.format(filetype))
    osImgDir= osImgDir.replace('\\', '/')
    osOutputDir = outputdir.replace('\\', '/')

    print(osImgDir+'\n', osOutputDir)

    #os.system('ffmpeg -r 24 -i "%s" -c:v libx264 -vf fps=25 -pix_fmt yuv420p %s' %(osImgDir, osOutputDir))
    subprocess.check_output('ffmpeg -r 24 -i "%s" -c:v libx264 -vf fps=25 -pix_fmt yuv420p %s' %(osImgDir, osOutputDir), 
        stderr=subprocess.STDOUT,
        shell=True)

    #test
    #subprocess.check_output('ffmpeg', stderr = subprocess.STDOUT, shell=True)
    
def openResizeCrop(imgdir, refsize):
    image = Image.open(imgdir)
    
    w,h = image.size

    if h!=refsize[1]:
        image = image.convert('RGB').resize( refsize, Image.ANTIALIAS)
        w,h = image.size
    
    if w > 2.5*h:
        image = image.crop( (w/2,0,w,h) )

    return(image)

#dirarr = [ ['110519/mid_cityscapes', '110519/mid'], ['110519/mid_cityscapes_mobile', '110519/mid_ade20k'] ]
def stackImages(dirarr, outputdir, refsize):
    """ stacks images according to 2d-list of directories and output in folder

    dir format has to use '\\'
    reference image height taken from first image in dirarr[0][0]
    doesn't fill the white space
    image names have to be numerical, starting from 0
    
    Args:
        dirarr: 2d-list according to stacking order     1 2
            e.g. [[dir1,dir2], [dir3,dir4]] results in  3 4
    """
    
    imgTotal = 0
    imgTotal = len([name for name in os.listdir(dirarr[0][0])]) #number of images
    if imgTotal == 0:
        print('nothing found in %s' %dirarr[0][0])

    for imgNo in range(imgTotal):
        if not os.path.exists( os.path.join(outputdir, '%s.png' %imgNo) ):

            print(os.path.join(outputdir, '%s.png' %imgNo))
            
            newimg = []
            
            for row in dirarr:
    
                #go through directories in rows, apply image adjustments according to reference size
                rowimgs = [ openResizeCrop( os.path.join(d, '%s.jpg' %imgNo), refsize ) for d in row ]
                newimg.append( np.hstack(rowimgs) )     #stack images horizontally
                
            stackedimg = np.vstack( newimg )
            temp = cv2.imwrite(os.path.join(outputdir, '%s.png' %imgNo), cvtColor(stackedimg, COLOR_RGB2BGR))
            #plt.imshow(stackedimg)
            #plt.axis('off')
            #plt.savefig(os.path.join(outputdir, '%s.png' %imgNo), bbox_inches='tight')
            #plt.clf()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        "--video_dir",
        default="",
        help= "path to folder with images")

    parser.add_argument(
        '-o',
        "--output_folder",
        default="",
        help= "path to folder for post-segmented images")
    
    parser.add_argument(
        '-r',
        "--fps",
        default="",
        type= int,
        help= "store image every x frame")
    
    parser.add_argument(
        '-f',
        "--frames_folder",
        default="",
        help= "path to folder for video frames")
    
    parser.add_argument(
        '-v',
        "--video_path",
        default="",
        help= "path for exported video")
    
    args = parser.parse_args()
    
    if args.video_dir != "" and args.output_folder != "":
        print('getting frames from video %s, outputting to %s, at %s fps' % (args.video_dir, args.output_folder, args.fps))
        getFrameFromVideo(args.video_dir, args.output_folder, args.fps)
    
    if args.frames_folder != "" and args.video_path != "":
        print('exporting video from %s as %s' % (args.frames_folder, args.video_path))
        getVideoFromFrame(args.frames_folder, args.video_path, 'png')