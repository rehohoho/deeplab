import os
from io import BytesIO
import tarfile
from six.moves import urllib
import argparse

# from matplotlib import gridspec
# from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import re
import glob

import tensorflow as tf
from post.post_utils import softmax_logits_forcrf, adhere_boundary
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from post.flagger_utils import imgGraph
# from skimage.segmentation import slic, mark_boundaries


CITYSCAPES_LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign',
    'vegetation', 'terrain',
    'sky',
    'person', 'rider',
    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
])

CITYSCAPES_TRANSDICT = {
    0:0,
    1:0,        #sidewalk: road
    2:2,
    3:2,        #wall: building
    4:2,        #fence
    5:2,        #pole
    6:2,        #traffic light
    7:2,        #traffic sign
    8:8,
    9:8,            #terrain: vegetation
    10:2,     #sky
    11:11,
    12:11,            #rider: person
    13:17,                #car: motorcycle
    14:17,                #truck
    15:17,                #bus
    16:17,                #train
    17:17,
    18:17                 #bicycle
}

CITYSCAPES_COLMAP = np.array( [
    [128, 64,128],    #road
    [244, 35,232],    #sidewalk
    [ 70, 70, 70],    #building
    [102,102,156],    #wall
    [190,153,153],    #fence
    [153,153,153],    #pole
    [250,170, 30],    #traffic light
    [220,220,  0],    #traffic sign
    [107,142, 35],    #vegetation
    [152,251,152],    #terrain
    [ 70,130,180],    #sky
    [220, 20, 60],
    [255,  0,  0],
    [  0,  0,142],
    [  0,  0, 70],
    [  0, 60,100],
    [  0, 80,100],
    [  0,  0,230],
    [119, 11, 32]
    
    ], dtype = np.uint8)


def write_tensors_to_txt(tarball_path, textfile_path):
    """ Create a text file with all tensors in checkpoint
    
    Args:
        tarball_path    path    to tar.gz containing frozen_inference_graph.pb
        textfile_path   path    to txt, to output
    Saves:
        txt file at textfile_path containing tensors from checkpoint
    """
    
    graph = tf.Graph()

    graph_def = None
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():    #find relevant graph and extract frozen graph into graph_def
        if 'frozen_inference_graph' in os.path.basename(tar_info.name):
            file_handle = tar_file.extractfile(tar_info)
            graph_def = tf.GraphDef.FromString(file_handle.read())
            break

    tar_file.close()

    with graph.as_default():                     #load graph def into self.graph
        tf.import_graph_def(graph_def)
    
    list_of_tuples = [op.values() for op in graph.get_operations()]
    list_of_tuples = [ str(i)+'\n' for i in list_of_tuples ]

    if textfile_path != '':
        f = open(textfile_path, 'w')
        f.write( ''.join(list_of_tuples) )
        f.close()


def showLabels(transdict, labelnames, colormap):
    """ show labels and corresponding colors in a plot
    
    Args:
        transdict:  dictionary      labels to translate to (eg. sky -> building)
        labelnames: np.array        labels for dataset (n_labels)
        colormap:   np.array        RGB colors for visualisation (n_labels x 3)
    
    FULL_LABEL_MAP = np.arange(len(CITYSCAPES_LABEL_NAMES)).reshape(len(CITYSCAPES_LABEL_NAMES), 1)
    FULL_COLOR_MAP = CITYSCAPES_COLMAP[FULL_LABEL_MAP] #same colors used for same labels
    """
    
    if transdict is not None:
        labels = [i for i in transdict.values()]
        unique_labels = np.unique(labels)
        new_colormap = [ FULL_COLOR_MAP[i] for i in unique_labels ]
    else:
        unique_labels = np.arange( len(labels) )
        new_colormap = FULL_COLOR_MAP
        
    grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6,1])
    
    ax = plt.subplot(grid_spec[0])
    plt.imshow(new_colormap, interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), labelnames[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    #SOFTMAX_TENSOR_NAME = 'ResizeBilinear_1:0'     #for cityscapes pretrained checkpoint
    SOFTMAX_TENSOR_NAME = 'ResizeBilinear_2:0'      #for custom checkpoint
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        
        self.graph = tf.Graph()

        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():  #find relevant graph and extract frozen graph into graph_def
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():           #load graph def into self.graph
            tf.import_graph_def(graph_def, name='')
        
        session_config = tf.ConfigProto()       #avoid cudnn load problem
        session_config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(graph=self.graph, config=session_config)

    def resize_image(self, image):
        
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        return(resized_image)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        
        resized_image = self.resize_image(image)
        
        batch_seg_map = self.sess.run(                    #run tf session to get mask
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        
        return resized_image, batch_seg_map[0]
    
    def getLogits(self, image):
        """Runs inference on a single image to get probabilities of each class

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image. (input_size, input_size, 3)
            logits: 2D array containing class probability of each pixel (pixelNo, classNo)
        """
        
        resized_image = self.resize_image(image)

        logits = self.sess.run(                    #run tf session to get mask
                self.SOFTMAX_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        
        logits = np.squeeze(logits)
        logits = logits[:target_size[1], :target_size[0]]
        
        return resized_image, logits


def numericalSort(filenames):
    """ sort filenames in directory by index, position, episode name
    ensure temporal continuity for post methods that use previous frames

    Args:
        filenames       list    of paths to images to perform inference
    
    Returns:
        new_filenames   list    sorted by index, then position (left, mid, right)
    """

    # check if filename has direction or is all digits
    filename = filenames[0]
    if not ('left' in filename or 'mid' in filename or 'right' in filename):
        new_filenames = filenames
        new_filenames.sort()
        return( new_filenames )

    new_filenames = []

    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            ind = re.findall(r'\d+', filename)  #index
            pos = None                          #left, mid, right
            if 'left' in filename:
                pos = 'left'
            if 'mid' in filename:
                pos = 'mid'
            if 'right' in filename:
                pos = 'right'
            episode_name = filename.split('/')[-2]
            new_filenames.append([filename, int(ind[-1]), pos, episode_name])
    
    new_filenames.sort(key=lambda x:x[1])
    new_filenames.sort(key=lambda x:x[2])
    new_filenames.sort(key=lambda x:x[3])
    new_filenames = [i[0] for i in new_filenames]

    return(new_filenames)


def parse_image_size(img_size_string):
    img_size = img_size_string.split(",")
    return (int(img_size[0]),int(img_size[1]))


def generate_CRFmodel(img, height, width, smLogits, n_labels, crf_pos, crf_col, crf_smooth):
    """ generate DenseCRF2D class for inference, for given image and logits

    - increasing crf gamma variables reduces the effect of original image on inference

    Args:
        img         3D numpy.array (height, width, 3) containing RGB channels
        height      int, height of image
        width       int, width of image
        smLogits    numpy.array (n_classes, ...) containing softmaxed logits
                    first dimension is the class, others are flattened
        n_labels    number of classes
        crf_pos     default 80  (higher better)
        crf_col     default 26 (lower better)
        crf_smooth  default 3
    """

    d = dcrf.DenseCRF2D(width, height, n_labels)

    U = unary_from_softmax(smLogits)
    d.setUnaryEnergy(U)
    
    # stddev x/y: position kernel (smoothness)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(crf_smooth, crf_smooth), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # stddev x/y: position kernel (appearance)
    # stddev rgb: color kernel (appearance)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(crf_pos, crf_pos), srgb=(crf_col, crf_col, crf_col), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    return(d)


def boundary_optimisation(image, n_segments, seg_map):
    """ TODO better boundary adherance using superpixels 
    
    1) get superpixels (SLIC, fast and good enough)
    2) for each superpixel, R[0..k]

            for each class in superpixel R[i], C[0..n]
                
                initialise wc = [0, n_pixels_in_class]
                initialise lc = [label, n_pixels_in_class]
                for pixel in pixels_in_class:
                    wc[pixel] = wc[pixel] +  1/n

    !!! adhere_boundary not done yet in post_utils.pyx
    """
    
    # SLIC: k-means clustering in RGB space
    im_superPix = slic(image, n_segments, sigma=5)
    im_adhereBoundary = adhere_boundary(im_superPix, seg_map)

    return(im_adhereBoundary)


def folder_seg(folder_path, output_path, 
                apply_crf, translate_labels, add_orig, vis, mark_main_road,
                crf_pos, crf_col, crf_smooth, mask_size):
    """ Segments all jpg from folderPath and outputs segmented image in outputpath
    
    Args:
        folder_path:        path    folder with images to segment
        output_path:        path    folder to output segmented images
        apply_crf           bool    whether to use crf
        translate_labels:   bool    whether to translate labels
        add_orig            bool    whether to attach original image
    """
    
    folderPath = os.path.abspath(folder_path)
    outputPath = os.path.abspath(output_path)

    if not os.path.exists(outputPath):                #creates directory if none found
        os.mkdir(outputPath)
        print('output directory not found, created one %s' %outputPath)
    
    filenames = [file for file in 
        glob.glob( os.path.join(folder_path, '**/*.png'), recursive=True )
    ]
    filenames = numericalSort(filenames)
    
    if vis:
        resize_method = Image.ANTIALIAS
    else:
        resize_method = Image.NEAREST

    # conventional segmentation, get confidence via probability spread, utilise previous frames, colorfulness metrics
    road_marker = None
    # for displaying transparency reflecting confidence, use post_utils to get confidence
    # black = Image.new('RGBA', (width*2, height), (0, 0, 0, 255))
    
    for filename in filenames:
        
        basename = os.path.join( filename.split('/')[-2], os.path.basename(filename) )
        
        # create directory for episode if doesn't exist
        dirname = os.path.join( output_path, filename.split('/')[-2] )
        if not os.path.exists(dirname):
            print('Creating directory at %s' %dirname)
            os.mkdir(dirname)

        if not os.path.exists( os.path.join(outputPath, basename) ):
            
            filedir = os.path.join(folderPath, filename)
            print(filedir)
            im = Image.open(filedir)
            
            # CRF requires logits, retrieve layer before argmax
            if apply_crf:
                resized_im, logits = MODEL.getLogits(im)
                height, width, classes = np.shape(logits)

                resized_im = np.array(resized_im)
                logits = softmax_logits_forcrf(height, width, logits)
                postcrf = generate_CRFmodel( resized_im, height, width, logits, 19, crf_pos, crf_col, crf_smooth)

                inf = postcrf.inference(5)          #(classes, pixels)
                seg_map = np.argmax(inf, axis = 0)  #(pixels)
                seg_map = np.reshape( seg_map, (height,width) )

            # retrieve argmaxed logits
            else:
                resized_im, seg_map = MODEL.run(im)

            if mark_main_road:
                if road_marker == None:      # handler only needs to initialised once
                    road_marker = imgGraph( np.shape(resized_im)[0], np.shape(resized_im)[1], \
                                            CITYSCAPES_COLMAP.astype(np.float32))
                
                road_marker.load_mask(seg_map.astype(np.uint8))
                road_marker.mark_main_road(0.5)   #threshold to reject road, percentage of height
                seg_map = road_marker.show_mask()

            if translate_labels:
                seg_map_len = len(seg_map)
                for i in range(seg_map_len):    #change mask index
                    seg_map[i] = [CITYSCAPES_TRANSDICT[j] for j in seg_map[i]]
            
            if vis:
                seg_image = CITYSCAPES_COLMAP[seg_map].astype(np.uint8)
            else:
                seg_image = seg_map.astype(np.uint8)

            if add_orig:
                seg_image = np.hstack( (resized_im, seg_image) )
            new_im = Image.fromarray(seg_image)

            if mask_size != new_im.size and not add_orig:
                new_im = new_im.resize(mask_size, resize_method)
                
            #Image.alpha_composite(black, new_im).save( os.path.join(outputPath, '%s.png' %filename[:-4]) )
            new_im.save( os.path.join(outputPath, basename) )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--image_folder',
        required=True,
        help= "path to folder with images"
    )
    parser.add_argument(
        '-o', '--output_folder',
        required=True,
        help= "path to folder for segmented images"
    ) 
    parser.add_argument(
        '-md', '--model_directory',
        required=True,
        help= "path to the directory with tar.gz model checkpoint"
    )
    parser.add_argument(
        '-cp', '--crf_pos',
        default=80,
        type=int,
        help= "position threshold for appearance kernel in crf pairwise potential"
    )
    parser.add_argument(
        '-cc', '--crf_col',
        default=26,
        type=int,
        help= "color threshold for appearance kernel in crf pairwise potential"
    )
    parser.add_argument(
        '-cs', '--crf_smooth',
        default=3,
        type=int,
        help= "position threshold for smoothing kernel in crf pairwise potential"
    )
    parser.add_argument(
        '-t', '--print_tensor',
        default='',
        help= "path to textfile with tensors"
    )
    parser.add_argument(
        '-p', '--use_crf',
        default=False,
        action='store_true',
        help= "apply post processing"
    )
    parser.add_argument(
        '-tl', '--translate_labels',
        default=False,
        action='store_true',
        help= "apply post processing"
    )
    parser.add_argument(
        '-ao', '--add_orig',
        default=False,
        action='store_true',
        help= "attach segmentation image with original"
    )
    parser.add_argument(
        '-v', '--vis_mask',
        default=False,
        action='store_true',
        help= "add visualization to mask"
    )
    parser.add_argument(
        '-gpu', '--gpu_utilise',
        default=0,
        type=int,
        help= "use the gpu"
    )
    parser.add_argument(
        '-mm', '--mark_main_road',
        default=False,
        action='store_true',
        help= "mark the main road"
    )
    parser.add_argument(
        '-is', '--mask_size',
        default="513,513",
        help= "change the image size to this tuple size"
    )
    args = parser.parse_args()

    if args.gpu_utilise == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disables GPU
    
    if not os.path.exists(args.model_directory):
        print('model directory not found. check is a tar.gz with frozen_inferene_graph.pb' %args.model_directory)
        exit

    print('model checkpoint found at %s, loading model...' %args.model_directory)
    MODEL = DeepLabModel(args.model_directory)
    
    if args.print_tensor != '':
        write_tensors_to_txt(args.model_directory, args.print_tensor)
        exit

    print('input folder %s, output folder %s' %(args.image_folder, args.output_folder))
    folder_seg(args.image_folder, args.output_folder, 
                args.use_crf, args.translate_labels, args.add_orig, args.vis_mask, args.mark_main_road,
                args.crf_pos, args.crf_col, args.crf_smooth,parse_image_size(args.mask_size))







"""
QUICK AND DIRTY TESTING
    requires files in terminal's working directory
        - frame_0_left.png (img)
        - frame_0_left.csv (logits)

import sys
sys.path.append('/home/whizz/data/data-engine')
from utils.post_utils import *
import numpy as np
from PIL import Image
from deeplab_example import *

im = Image.open('frame_0_left.png')
im = im.convert('RGB').resize( (513,225), Image.ANTIALIAS)
im = np.array(im)
height, width = np.shape(im)[:2]
post = imgGraph(height, width)
logits = np.loadtxt(open("frame_0_left.csv", "r"), dtype=np.float32, delimiter=",")

        # crf 

raw_logits = np.reshape(np.copy(logits), (225,513,19) )

seg_map = argmax_logits(height, width, raw_logits)
im_bo = boundary_optimisation(im, seg_map, 200)

nll, conf, n, colorfulness = calc_confidence_forcrf( 0.9, 0.1, 225, 513, raw_logits, im)
postcrf = generate_CRFmodel( im, height, width, nll, 19, 80, 26, 3)
inf = postcrf.inference(5)          #(classes, pixels)
seg_map = np.argmax(inf, axis = 0)  #(pixels)
seg_map = np.reshape( seg_map, (height,width) )
seg_image = CITYSCAPES_COLMAP[seg_map].astype(np.uint8)
seg_image = Image.fromarray(seg_image)
seg_image.save('frame_0_left_segpost_withcrf.png')

post.load_logits(logits)
post.softmax_logits()
post.no_post()
post.calc_confidence(0.1)
conf = post.show_confidence()

        # get image with confidence as alpha value

rawim = Image.fromarray(post.show_image(1))
height, width = np.shape(rawim)[:2]
#rawim.save('frame_0_left_seg.png')
black = Image.new('RGBA', (width, height), (0, 0, 0, 255))
Image.alpha_composite(black, im).save('frame_0_left_seg.png')

        # consider 1 frame back, weight by confidence

post.calc_confidence(0.1)
post.post(1.5) 
post.pushback_logits()
rawim = Image.fromarray(post.show_image(1))...

        # conventional segmenting

post.load_rgb(im)
post.calc_edges()
post.oversegment(0.02)
np.savetxt( 'bloop.csv' , post.show_parents(), delimiter=",")
postim = Image.fromarray(post.show_image(1))
postim.save('frame_0_left_segpost.png')

        # plot confidence histogram

import matplotlib.pyplot as plt
plt.hist(conf2[0].flatten(), bins=30)
plt.show()

more ideas:
- check color histogram, if colours less spread out increase crf col threshold (2x)
- check segmentation confidence, increase crf thresholds (appearance accordingly)
- compare to average confidence?


import glob
import os
import numpy as np
from PIL import Image
files = [f for f in glob.glob('**/*.png')]
print(len(files))
for f in files:
    classes = np.unique(np.array(Image.open(f)))
    if not np.all(classes<20):
        print(f, classes)
"""