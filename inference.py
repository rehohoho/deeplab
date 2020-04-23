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
from post.post_utils import adhere_boundary, softmax_logits_forcrf
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
    LOGITS_TENSOR_NAME = 'ResizeBilinear_2:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path, mask_size):
        """Creates and loads pretrained deeplab model."""
        
        self.graph = tf.Graph()
        self.mask_max_side = max(mask_size)

        # Retrieve graph definition
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        # Setup up graph and session
        with self.graph.as_default():
            
            tf.import_graph_def(graph_def, name='')
            max_side_size = max(mask_size)  # process resize within graph, preserves aspect ratio

            logits = tf.get_default_graph().get_tensor_by_name(self.LOGITS_TENSOR_NAME)
            self.logits = tf.image.resize(tf.squeeze(logits), size=[max_side_size, max_side_size])
            
        session_config = tf.ConfigProto()   # avoid cudnn load problem
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=session_config)

    def resize_image(self, image, tar_max_size):
        """ Resize image such that longer size is tar_max_size """

        width, height = image.size
        resize_ratio = tar_max_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_im = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        return resized_im

    def get_logits_layer(self, image):
        """Runs inference on a single image to get probabilities of each class

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image. (input_size, input_size, 3)
            result: 2D array containing class probability of each pixel (pixelNo, classNo)
        """
        
        output_image = self.resize_image(image, self.mask_max_side)
        model_input = self.resize_image(image, self.INPUT_SIZE)

        result = self.sess.run(
                self.logits, # logits layer have bilinear scale to size of output, see __init__
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(model_input)]})
        
        # get relevant portion of logits
        width, height = output_image.size
        result = result[:height, :width]
        result = result.transpose(2, 0, 1)

        return output_image, result


def generate_CRFmodel(img, height, width, smLogits, n_labels, crf_config):
    """ generate DenseCRF2D class for inference, for given image and logits

    - increasing crf gamma variables reduces the effect of original image on inference

    Args:
        img         3D numpy.array (height, width, 3) containing RGB channels
        height      int, height of image
        width       int, width of image
        smLogits    numpy.array (n_classes, ...) containing softmaxed logits
                    first dimension is the class, others are flattened
        n_labels    number of classes
        crf_config  default '80,26,3'
    """
    crf_pos, crf_col, crf_smooth = [int(i) for i in crf_config.split(',')]
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


def generate_directory_path(path):

    if os.path.exists(path):
        return
    
    generate_directory_path( os.path.dirname(path) )
    os.mkdir(path)
    print('created directory at %s' %path)


def folder_seg(folder_path, output_path, 
                apply_crf, translate_labels, add_orig, vis, mark_main_road, output_logits,
                crf_config):
    """ Segments all jpg from folderPath and outputs segmented image in outputpath
    
    Args:
        folder_path:        path    folder with images to segment
        output_path:        path    folder to output segmented images
        apply_crf           bool    flag to use crf
        translate_labels:   bool    flag to translate labels
        add_orig            bool    flag to attach original image
        vis                 bool    flag to visualise n_classes
        mark_main_road      bool    flag to find classes connecting to pavement at middle-bottom
        output_logits       bool    flag to output logits only
    """
    
    folderPath = os.path.abspath(folder_path)
    
    filenames = [file for file in 
        glob.glob( os.path.join(folder_path, '**/*.png'), recursive=True )
    ]
    
    if vis:
        resize_method = Image.ANTIALIAS
    else:
        resize_method = Image.NEAREST

    road_marker = None  # for conventional segmentation, get confidence via probability spread, utilise previous frames, colorfulness metrics
    # black = Image.new('RGBA', (width*2, height), (0, 0, 0, 255))  # for displaying transparency reflecting confidence, use post_utils to get confidence
    
    for filename in filenames:
        
        save_path = filename.replace(folder_path, output_path)
        if output_logits:
            save_path = '%s.npy' %os.path.splitext(save_path)[0]
        generate_directory_path(os.path.dirname(save_path))

        if not os.path.exists(save_path):
            
            filedir = os.path.join(folderPath, filename)
            print('%s -> %s' %(filedir, save_path))
            im = Image.open(filedir)
            
            resized_im, logits_layer = MODEL.get_logits_layer(im)
            classes, height, width = np.shape(logits_layer)

            if apply_crf:
                softmax_layer = softmax_logits_forcrf(height, width, logits_layer.transpose(1,2,0))
                postcrf = generate_CRFmodel(np.array(resized_im), height, width, softmax_layer.copy(order='C'), 19, crf_config)
                logits_layer = postcrf.inference(5)          #(classes, pixels)
            
            if output_logits:
                logits_layer = np.array(logits_layer, dtype = np.float16).reshape((19, height, width))
                np.save(save_path, logits_layer)

            else:
                seg_map = np.argmax(logits_layer, axis=0).reshape((height,width)).astype(np.uint8)

                if mark_main_road:
                    if road_marker == None:      # handler only needs to initialised once
                        road_marker = imgGraph( np.shape(resized_im)[0], np.shape(resized_im)[1], \
                                                CITYSCAPES_COLMAP.astype(np.float32))
                    
                    road_marker.load_mask(seg_map)
                    road_marker.mark_main_road(0.5)   #threshold to reject road, percentage of height
                    seg_map = road_marker.show_mask()

                if translate_labels:
                    seg_map_len = len(seg_map)
                    for i in range(seg_map_len):    #change mask index
                        seg_map[i] = [CITYSCAPES_TRANSDICT[j] for j in seg_map[i]]
                
                if vis:
                    seg_map = CITYSCAPES_COLMAP[seg_map].astype(np.uint8)
                
                if add_orig:
                    seg_map = np.hstack( (resized_im, seg_map) )
                new_im = Image.fromarray(seg_map)

                #Image.alpha_composite(black, new_im).save( os.path.join(outputPath, '%s.png' %filename[:-4]) )
                new_im.save(save_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Model and data paramters
    parser.add_argument(
        '--image_folder',
        required=True,
        help= "path to folder with images"
    )
    parser.add_argument(
        '--output_folder',
        required=True,
        help= "path to folder for segmented images"
    ) 
    parser.add_argument(
        '--model_directory',
        required=True,
        help= "path to the directory with tar.gz model checkpoint"
    )
    parser.add_argument(
        '--mask_size',
        default="513,513",
        help= "width, height of image size (eg. 513,513)"
    )
    parser.add_argument(
        '--output_logits',
        default=False,
        action='store_true',
        help= "flag to output logits instead of image data"
    )
    # Post-processing arguments
    parser.add_argument(
        '--crf_config',
        default='80,26,3',
        help= "crf pairwise potential kernal config (position, color for appearance, position for smoothing)"
    )
    parser.add_argument(
        '--use_crf',
        default=False,
        action='store_true',
        help= "flag to apply crf"
    )
    parser.add_argument(
        '--mark_main_road',
        default=False,
        action='store_true',
        help= "flag to mark the main road"
    )
    # Visualisation flags
    parser.add_argument(
        '--vis_mask',
        default=False,
        action='store_true',
        help= "flag to turn mask into visualisation"
    )
    parser.add_argument(
        '--add_orig',
        default=False,
        action='store_true',
        help= "flag to attach segmentation image with original image"
    )
    parser.add_argument(
        '--translate_labels',
        default=False,
        action='store_true',
        help= "flag to translate labels (e.g. sky to building)"
    )
    # Others
    parser.add_argument(
        '--print_tensor_path',
        default='',
        help= "path to textfile with tensors"
    )
    parser.add_argument(
        '--use_cpu',
        default=False,
        action='store_true',
        help= "use CPU only"
    )
    args = parser.parse_args()

    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disables GPU
    
    if not os.path.exists(args.model_directory):
        print('model directory not found. check is a %s tar.gz with frozen_inferene_graph.pb' %args.model_directory)
        exit
    
    print('model checkpoint found at %s, loading model...' %args.model_directory)
    args.mask_size = args.mask_size.split(',')
    args.mask_size = [int(i) for i in args.mask_size]
    MODEL = DeepLabModel(args.model_directory, args.mask_size)
    
    if args.print_tensor_path != '':
        write_tensors_to_txt(args.model_directory, args.print_tensor_path)
    else:
        print('input folder %s, output folder %s' %(args.image_folder, args.output_folder))
        folder_seg(args.image_folder, args.output_folder, 
                args.use_crf, args.translate_labels, args.add_orig, args.vis_mask, args.mark_main_road, args.output_logits,
                args.crf_config)







"""
NPY TO VIS
filename = 'D:/perception_datasets/scooter_halflabelled/scooter_images/smthelse/train/frame_444_mid.npy'
logits = np.load(filename)
seg_map = np.argmax(logits, axis=0).astype(np.uint8)
seg_map = CITYSCAPES_COLMAP[seg_map]
Image.fromarray(seg_map).save(filename.replace('npy', 'png'))
"""

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