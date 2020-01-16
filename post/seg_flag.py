import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os

from post.flagger_utils import *
import argparse


CITYSCAPES_TRANSDICT = {
    0:0,
    1:0,    #sidewalk: road
    2:2,
    3:2,    #wall: building
    4:2,    #fence
    5:2,    #pole
    6:2,    #traffic light
    7:2,    #traffic sign
    8:8,
    9:8,      #terrain: vegetation
    10:2,   #sky
    11:11,
    12:11,      #rider: person
    13:17,        #car: motorcycle
    14:17,        #truck
    15:17,        #bus
    16:17,        #train
    17:17,
    18:17         #bicycle
}


def create_cityscapes_colormap():
    """ Creates a label colormap used in CITYSCAPES segmentation benchmark.

      Returns:
        A Colormap for visualizing segmentation results.
    """
    
    return( np.array( [
    [128, 64,128],  #road
    [244, 35,232],  #sidewalk
    [ 70, 70, 70],  #building
    [102,102,156],  #wall
    [190,153,153],  #fence
    [153,153,153],  #pole
    [250,170, 30],  #traffic light
    [220,220,  0],  #traffic sign
    [107,142, 35],  #vegetation
    [152,251,152],  #terrain
    [ 70,130,180],  #sky
    [220, 20, 60],
    [255,  0,  0],
    [  0,  0,142],
    [  0,  0, 70],
    [  0, 60,100],
    [  0, 80,100],
    [  0,  0,230],
    [119, 11, 32]
    
    ], dtype = np.uint8) )


def create_truncated_cityscapes_colormap():
    """ Creates a label colormap used in CITYSCAPES segmentation benchmark.

      Returns:
        A Colormap for visualizing segmentation results.
    """
    
    return( np.array( [
    [128, 64,128],  #road
    [255,255,255],  #sidewalk
    [ 70, 70, 70],  #building
    [255,255,255],  #wall
    [255,255,255],  #fence
    [255,255,255],  #pole
    [255,255,255],  #traffic light
    [255,255,255],  #traffic sign
    [107,142, 35],  #vegetation
    [255,255,255],  #terrain
    [255,255,255],  #sky
    [220, 20, 60],
    [255,255,255],
    [255,255,255],
    [255,255,255],
    [255,255,255],
    [255,255,255],
    [  0,  0,230],
    [255,255,255]
    
    ], dtype = np.uint8) )
    
    
def circular_kernel(a, b, n, r):
    """ outputs 2D array with circular kernel at position (a,b) of radius r of mask size n

    Args:
        a, b: (x,y) coordinate denoting centre of circle
        r: radius of circle
        n: size of mask
    """
    
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array = np.ones((n, n))
    array[mask] = 255

    return(array)


def opt_circular_kernel(imgPath, outputPath, radiusStart, radiusEnd):
    """ outputs denoised segmentation for multiple circular kernels 
    
    Args:
        imgPath:        string,     path to image
        outputPath:     string,     path to output image
        radiusStart:    int (odd),  min radius for circular kernel
        radiusEnd:      int (odd),  max radius for circular kernel
    """
    
    rawim = Image.open(imgPath)
    im = np.array(rawim)
    height, width = im.shape[:2]
    im = im[0:height, int(width/2):width]
    
    for i in range(3,21,2):
        for j in range(1,int(i/2)+1):
            kernel = circular_kernel( int(i/2), int(i/2), i, j)
            print(kernel)
            openim = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
            openim = Image.fromarray(openim)
            openim.save( os.path.join(outputPath,'size%s_radius%s.png' %(i,j)) )


def label_to_color_image(label, colormap):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def alter_and_flag_img(inputPath, outputPath, cm, outputcm):
    """ output altered segmentation and print flags

    Args:
        inputPath:  string,     path to image
        outputPath: string,     directory for altered segmentation output
        cm:         2D array (labels, 3)    colormap used for initial segmentation
        outputcm:   2D array (labels, 3)    colormap used for output
    """
    
    rawim = Image.open(inputPath)
    im = np.array(rawim)
    height, width = im.shape[:2]
    im = im[0:height, int(width/2):width]
    
    kernel = circular_kernel( 6, 6, 11, 4)
    im = cv2.dilate(im,kernel,iterations = 1)
    im = cv2.erode(im,kernel,iterations = 1)
    
    height, width = im.shape[:2]
    test = imgGraph(height, width, cm)  #remove, initialise test outside: labels, colormap

    test.load_labels(im)                #overwrite labels and update buckets
    test.mark_main_road()               #overwrite labels and update buckets

    mask = test.show_mask()
    buckets = test.show_buckets()
    seg = label_to_color_image(mask, outputcm)
    seg = Image.fromarray(seg)
    seg.save( '%s.png' %os.path.splitext(outputPath)[0] )
    
    flags = test.show_flags(0.75, 0.5, 0, True)
    
    print(flags)


def alter_and_flag_folder(imgFolder, outputFolder, trashFolder, wrong, minsize, maxwidth, showMetrics):
    """ output altered segmentation and print flags
    
    1) get flag based on raw segmentation (no merge, no denoise, no main road)
    2) post process and output image
    
    e.g. alter_and_flag_folder('data\\230419_mid_cityscapes', 'data\\230419_mid_post', trun_cm_32, cm, 0.15, 0.6, 0.7, False)
    
    Args:
        imgFolder:      string,                 path to folder with images
        outputFolder:   string,                 path for altered segmentations
        trashFolder:    string,                 path for flagged altered segmentations
        showMetrics:    bool,                   include flagging metrics in csv
    """
    
    cm = create_cityscapes_colormap()           #generate cityscapes colormaps
    trun_cm = create_truncated_cityscapes_colormap()
    trun_cm_32 = create_truncated_cityscapes_colormap().astype(np.float32)
    
    inputPath = os.path.abspath(imgFolder)
    outputPath = os.path.abspath(outputFolder)
    trashPath = os.path.abspath(trashFolder)
    
    if not os.path.exists(outputPath):          #creates directory if none found
        os.mkdir(outputPath)
        print('output directory not found, created one %s' %outputPath)
    if not os.path.exists(trashPath):          #creates crap directory if none found
        os.mkdir(trashPath)
        print('flagged output directory not found, created one %s' %trashPath)
    
    graph = None                                #initialise graph object and morphology kernel
    kernel = circular_kernel( 6, 6, 11, 4)

    with open( os.path.join(outputPath, 'probably_crap.csv'), 'w') as txtOutput:
    
        filenames = os.listdir(inputPath)
        for filename in filenames:
            if filename.endswith('png') and not os.path.exists( os.path.join(outputPath, filename) ):
                
                imgPath = os.path.join(inputPath, filename)     #gets image as numpy array
                rawim = Image.open(imgPath)
                im = np.array(rawim)

                height, width = im.shape[:2]
                im_cropped = im[0:height, int(width/2):width]   #image crop (remove left half showing orig)
                
                if graph == None:                   #initialise graph using size of first image
                    height, width = im_cropped.shape[:2]
                    print(height, width)
                    graph = imgGraph(height, width, trun_cm_32)
                else:                               #reset buckets only otherwise
                    graph.reset_bucketsAndFlags()
                
                graph.load_labels(im_cropped)       #get flags from raw segmentation before morphing
                graph.mark_main_road(minsize)       #mark out road connecting to bottom as pavement
                
                flags = graph.show_flags(wrong, minsize, maxwidth)  #0.6, 0.5, 0.9
                
                if showMetrics:
                    txtOutput.write( '%s, %s\n' %(filename, flags) )
                else:
                    if flags[0] != ',':
                        txtOutput.write( '%s, %s\n' %(filename, flags.split(',')[0]) )
                
                #""" saving image
                im_morphed = cv2.dilate(im_cropped,kernel,iterations = 1)
                im_morphed = cv2.erode(im_morphed,kernel,iterations = 1)
                im_morphed = cv2.morphologyEx(im_morphed, cv2.MORPH_OPEN, kernel)
                graph.load_labels(im_morphed)         #output morphed image
                graph.mark_main_road(minsize)
                mask = graph.show_mask()
                
                seg = label_to_color_image(mask, cm)  #color mask using output colormap
                new_seg = np.hstack( (im, seg) )      #append to original segmentation
                
                new_seg = Image.fromarray(new_seg)
                if flags[0] != ',':
                    new_seg.save( '%s.png' %os.path.splitext(os.path.join(trashPath, filename))[0] )
                else:
                    new_seg.save( '%s.png' %os.path.splitext(os.path.join(outputPath, filename))[0] )


if __name__ == '__main__':    
    
    trun_cm = create_truncated_cityscapes_colormap()
    cm = create_cityscapes_colormap()
    trun_cm_32 = trun_cm.astype(np.float32)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-i',
        "--input_folder",
        required=True,
        help= "path to folder with images")

    parser.add_argument(
        '-o',
        "--output_folder",
        required=True,
        help= "path to folder for post-segmented images")
    
    parser.add_argument(
        '-t',
        "--trash_folder",
        required=True,
        help= "path to folder for flagged post-segmented images")
    
    parser.add_argument(
        '-f1',
        "--road_wrongness",
        default= 0.6,
        help= "threshold for wrong road (0-1)")
    
    parser.add_argument(
        '-f2',
        "--road_minimum_size",
        default= 0.5, #~200
        help= "min threshold for road boundary size (pixels)")
    
    parser.add_argument(
        '-f3',
        "--road_maximum_width",
        default= 0.9,
        help= "max threshold for road width wrt image width")
    
    args = parser.parse_args()
    alter_and_flag_folder( args.input_folder, args.output_folder, args.trash_folder,
        args.road_wrongness, args.road_minimum_size, args.road_maximum_width, True)