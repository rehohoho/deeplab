import os
from io import BytesIO
import tarfile
from six.moves import urllib

# from matplotlib import gridspec
# from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import re

import tensorflow as tf
from post.post_utils import *
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from skimage.segmentation import slic, mark_boundaries


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

def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.

        Returns:
            A Colormap for visualizing segmentation results.
    """
    
    return( np.array( [
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
    
    ], dtype = np.uint8) )


def print_tensors(tarball_path, textfile_path):
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
    
    return()


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    #SOFTMAX_TENSOR_NAME = 'ResizeBilinear_1:0'     #for cityscapes checkpoint
    SOFTMAX_TENSOR_NAME = 'ResizeBilinear_2:0'      #for custom checkpoint
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        
        self.graph = tf.Graph()

        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():    #find relevant graph and extract frozen graph into graph_def
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():             #load graph def into self.graph
            tf.import_graph_def(graph_def, name='')
        
        #avoid cudnn load problem
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(graph=self.graph, config=session_config)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        
        width, height = image.size                            #resize image
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        
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
        width, height = image.size                            #resize image
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        
        logits = self.sess.run(                    #run tf session to get mask
                self.SOFTMAX_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        
        logits = np.squeeze(logits)
        logits = logits[:target_size[1], :target_size[0]]
        
        return resized_image, logits


def label_to_color_image(label):
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

    #colormap = create_pascal_label_colormap()
    colormap = create_cityscapes_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def showLabels(transdict, labelnames, colormap):
    """ show labels and corresponding colors in a plot
    
    Args:
        transdict: dictionary with labels to translate to
        labelnames: (np)array with labels
        colormap: 2D-(np)array with RGB colors
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


def folder_seg(fp, op, translateMaskInd, comparison):
    """ Segments all jpg from folderPath and outputs segmented image in outputpath
    
    Args:
        fp: path of folder with images to segment
        op: path of folder to output segmented images
        translateMaskInd: bool to indicate translation of index
    """
    
    folderPath = os.path.abspath(fp)
    outputPath = os.path.abspath(op)

    if not os.path.exists(outputPath):          #creates directory if none found
        os.mkdir(outputPath)
        print('output directory not found, created one %s' %outputPath)
    
    for filename in os.listdir(folderPath):
        if (filename.endswith('jpg') or filename.endswith('png')) and not os.path.exists( os.path.join(outputPath, '%s.png' %filename[:-4]) ):
            
            filedir = os.path.join(folderPath, filename)
            print(filedir)
            
            im = Image.open(filedir)
            resized_im, seg_map = MODEL.run(im)
            
            if translateMaskInd:
                seg_map_len = len(seg_map)
                for i in range(seg_map_len):    #change mask index
                    seg_map[i] = [CITYSCAPES_TRANSDICT[j] for j in seg_map[i]]

            seg_image = label_to_color_image(seg_map).astype(np.uint8)
                            
            if comparison:
                seg_image = np.hstack( (resized_im, seg_image) )
            new_im = Image.fromarray(seg_image)
            new_im.save( os.path.join(outputPath, '%s.png' %filename[:-4]) )


def numericalSort(filenames):
    """ sort filenames in directory by index, then position
    ensure temporal continuity for post methods that use previous frames
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


def folder_seg_withpost(fp, op, translateMaskInd, crf_pos, crf_col, crf_smooth, comparison):
    """ Segments all jpg from folderPath and outputs segmented image in outputpath
    
    Args:
        fp: path of folder with images to segment
        op: path of folder to output segmented images
        translateMaskInd: bool to indicate translation of index
    """
    
    folderPath = os.path.abspath(fp)
    outputPath = os.path.abspath(op)

    if not os.path.exists(outputPath):                #creates directory if none found
        os.mkdir(outputPath)
        print('output directory not found, created one %s' %outputPath)
    
    filenames = os.listdir(folderPath)
    filenames = numericalSort(filenames)
    post = None             # my trash
    postcrf = None          # crf model
    alph = np.array([])     # for displaying transparency reflecting confidence
    black = np.array([])
    outputCSV = []
    
    for filename in filenames:
        if not os.path.exists( os.path.join(outputPath, '%s.png' %filename[:-4]) ):
            
            filedir = os.path.join(folderPath, filename)
            print(filedir)
            
            im = Image.open(filedir)
            resized_im, logits = MODEL.getLogits(im)
            height, width, classes = np.shape(logits)
            
            if post == None:
                post = imgGraph( np.shape(resized_im)[0], np.shape(resized_im)[1] )
            if alph.size == 0:
                alph = np.full( (height,width,1), 255, dtype = np.uint8 )
            if black.size == 0:
                black = Image.new('RGBA', (width*2, height), (0, 0, 0, 255))

            #logits = softmax_logits_forcrf(height, width, logits)
            resized_im = np.array(resized_im)
            logits, conf, n, colorfulness = calc_confidence_forcrf( 0.9, 0.3, height, width, logits, resized_im)
            postcrf = generate_CRFmodel( resized_im, height, width, logits, 19, crf_pos, crf_col, crf_smooth)

            inf = postcrf.inference(5)          #(classes, pixels)
            seg_map = np.argmax(inf, axis = 0)  #(pixels)
            seg_map = np.reshape( seg_map, (height,width) )
            seg_image = label_to_color_image(seg_map).astype(np.uint8)

            if comparison:
                seg_image = np.hstack( (resized_im, seg_image) )
            new_im = Image.fromarray(seg_image)
            #Image.alpha_composite(black, new_im).save( os.path.join(outputPath, '%s.png' %filename[:-4]) )
            new_im.save( os.path.join(outputPath, '%s.png' %filename[:-4]) )

FULL_LABEL_MAP = np.arange(len(CITYSCAPES_LABEL_NAMES)).reshape(len(CITYSCAPES_LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP) #same colors used for same labels


#https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md to check specs
_MODEL_URLS = {
    #PASCAL VOC 2012
        'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',

    #CITYSCAPES
        #no ImageNet, stride 32, scale [1.0], mIOU 72.41 (val), multadds 15.95B
        'mobilenetv3_large_cityscapes_trainfine':
                'deeplab_mnv3_large_cityscapes_trainfine_2019_11_15.tar.gz',
        #no ImageNet, stride 32, scale [1.0], mIOU 68.99, multadds 4.63B
        'mobilenetv3_small_cityscapes_trainfine':
                'deeplab_mnv3_small_cityscapes_trainfine_2019_11_15.tar.gz',
        #stride 16,8, scale [1.0], no/yes flip, [0.75,0.25,1.25], mIOU 78.79/80.42 (val), multadds 419B/8678B
        'xception65_cityscapes_trainfine':
                'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
        #stride 16, scale [1.0], no flip, mIOU 80.31 (val), multadds 502B -> current best
        'xception71_dpc_cityscapes_trainfine':
                'deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz',
        #stride 8, scale [0.75,0.25,2], flip, mIOU 82.66 (test) -> my computer cannot handle
        'xception71_dpc_cityscapes_trainval':
                'deeplab_cityscapes_xception71_trainvalfine_2018_09_08.tar.gz',

    #ADE20K
        #stride 16, scale [1.0], no flip, mIOU 32.04% (val), pixelacc 75.41% (val)
        'mobilenetv2_ade20k_train':
                'deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz',
        #stride 8, scale [0.25,0.5,1.75], flip, mIOU 45.65% (val), pixelacc 82.52% (val)
        'xception65_ade20k_train':
                'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'
}

"""
MODEL_NAME = 'xception71_dpc_cityscapes_trainfine' #mobilenetv2_ade20k_train
model_dir = os.getcwd() + '\\pretrained_models'
download_path = os.path.join(model_dir, MODEL_NAME+'.tar.gz')
print('MODEL = DeepLabModel(download_path)')
print("folder_seg('230419/mid', '230419/mid_cityscapes', True, '')")
#"""
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'


import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        "--image_folder",
        required=True,
        help= "path to folder with images")

    parser.add_argument(
        '-o',
        "--output_folder",
        required=True,
        help= "path to folder for segmented images")
    
    parser.add_argument(
        '-md',
        "--model_directory",
        required=True,
        help= "path to the directdory with tar.gz model checkpoint")

    parser.add_argument(
        '-mv',
        "--model_variant",
        default= 'xception71_dpc_cityscapes_trainfine',
        help= "variant of model")
    
    parser.add_argument(
        '-cp',
        "--crf_pos",
        default=80,
        help= "position threshold for appearance kernel in crf pairwise potential")
    
    parser.add_argument(
        '-cc',
        "--crf_col",
        default=26,
        help= "color threshold for appearance kernel in crf pairwise potential")
    
    parser.add_argument(
        '-cs',
        "--crf_smooth",
        default=3,
        help= "position threshold for smoothing kernel in crf pairwise potential")
    
    parser.add_argument(
        '-t',
        "--print_tensor",
        default='',
        help= "path to textfile with tensors")

    parser.add_argument(
        '-p',
        "--post",
        default=1,
        type= int,
        help= "apply post processing")

    parser.add_argument(
        '-c',
        "--comparison",
        default=1,
        type= int,
        help= "attach segmentation image with original")
    
    parser.add_argument(
        '-gpu',
        "--gpu_utilise",
        default=0,
        help= "use the gpu")

    args = parser.parse_args()

    if args.gpu_utilise == "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disables GPU

    download_path = os.path.join(args.model_directory, args.model_variant+'.tar.gz')
    
    if not os.path.exists(args.model_directory):
        print('model directory not found. creating directory %s' %download_path)
        os.mkdir(args.model_directory)

    if not os.path.exists(download_path):
        download_url = _DOWNLOAD_URL_PREFIX + _MODEL_URLS[args.model_variant]
        print('no model found, downloading model from %s into %s' %(download_url, download_path))
        urllib.request.urlretrieve(download_url, download_path)

    print('model checkpoint found at %s, loading model...' %download_path)
    MODEL = DeepLabModel(download_path)
    
    if args.print_tensor != '':
        print_tensors(download_path, args.print_tensor)

    print('input folder %s, output folder %s' %(args.image_folder, args.output_folder))
    if args.post:
        print('applying post processing')
        folder_seg_withpost(args.image_folder, args.output_folder, True, 
                            int(args.crf_pos), int(args.crf_col), int(args.crf_smooth),
                            args.comparison)
    else:
        print('not applying post processing')
        folder_seg(args.image_folder, args.output_folder, True, args.comparison)






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
seg_image = label_to_color_image(seg_map).astype(np.uint8)
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
"""