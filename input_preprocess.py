# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Prepares the data used for DeepLab training/evaluation."""
import tensorflow as tf
from deeplab.core import feature_extractor
from deeplab.core import preprocess_utils

import numpy as np
from random import random
from math import radians

# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5


def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               is_training=True,
                               model_variant=None,
                               motion_blur_size=0,
                               motion_blur_direction_limit=30,
                               rotation_min_limit=0,
                               rotation_max_limit=0,
                               brightness_min_limit=0,
                               brightness_max_limit=0):
  """Preprocesses the image and label.

  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  original_image = image
  processed_image = tf.cast(image, tf.float32)
  
  # CUSTOM AUGMENTATION 2: Rotation in degrees
  if rotation_min_limit != 0:
    rotation_deg = (rotation_max_limit - rotation_min_limit) * random() + rotation_min_limit
    processed_image = tf.contrib.image.rotate(processed_image, radians(rotation_deg))

    #in-built rotate function zeroes all empty spaces, add one to all classes so class 0 is not affected
    if label is not None:
      label = tf.math.add(label, tf.ones_like(label))
      label = tf.contrib.image.rotate(label, radians(rotation_deg))
      
      #ensure label is dtype=uint8, for 0-1=255, since 255 is ignore label
      label = tf.math.subtract(label, tf.ones_like(label))
  
  if label is not None:
    label = tf.cast(label, tf.int32)
  
  # Resize image and label to the desired range.
  if min_resize_value or max_resize_value:
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))
    # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)
  
  # CUSTOM AUGMENTATION 1: Add motion blur, kernel wrt height, directoin in degrees
  if motion_blur_size != 0:
      processed_image = add_blur(processed_image, crop_height, motion_blur_size, motion_blur_direction_limit)
      processed_image.set_shape([None, None, 3])
  
  # CUSTOM AUGMENTATION 3: Add brightness, in multiple of original brightness
  if brightness_min_limit != 0:
      brightnessLevel = (brightness_max_limit - brightness_min_limit) * random() + brightness_min_limit
      processed_image = tf.image.adjust_brightness(processed_image, brightnessLevel)
  
  # Data augmentation by randomly scaling the inputs.
  if is_training:
    scale = preprocess_utils.get_random_scale(
        min_scale_factor, max_scale_factor, scale_factor_step_size)
    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, scale)
    processed_image.set_shape([None, None, 3])
  
  # Pad image and label to have dimensions >= [crop_height, crop_width]
  image_shape = tf.shape(processed_image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  target_height = image_height + tf.maximum(crop_height - image_height, 0)
  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  # Pad image with mean pixel value.
  mean_pixel = tf.reshape(
      feature_extractor.mean_pixel(model_variant), [1, 1, 3])
  processed_image = preprocess_utils.pad_to_bounding_box(
      processed_image, 0, 0, target_height, target_width, mean_pixel)

  if label is not None:
    label = preprocess_utils.pad_to_bounding_box(
        label, 0, 0, target_height, target_width, ignore_label)

  # Randomly crop the image and label.
  if is_training and label is not None:
    processed_image, label = preprocess_utils.random_crop(
        [processed_image, label], crop_height, crop_width)

  processed_image.set_shape([crop_height, crop_width, 3])

  if label is not None:
    label.set_shape([crop_height, crop_width, 1])

  if is_training:
    # Randomly left-right flip the image and label.
    processed_image, label, _ = preprocess_utils.flip_dim(
        [processed_image, label], _PROB_OF_FLIP, dim=1)

  return original_image, processed_image, label


def createKernel(direction, size):
    """ create motion blur kernel

    Args:
        direction:  float, degrees specifying direction of blur, counter-clockwise positive
        size:       int, size of kernel specifying extent of blur
    """

    assert(size%2 == 1), 'size needs to be positive odd integer'
    
    k = np.zeros( (size, size), dtype=np.float32 )
    
    unit_vector = np.tan( direction*np.pi / 180 )
    
    centre = int(size/2)
    rad = centre + 1

    if abs(unit_vector) < 1:
        for xshift in range(rad):
            yshift = np.round(unit_vector * xshift)
            
            k[int(centre - yshift), centre + xshift] = 1
            k[int(centre + yshift), centre - xshift] = 1
    else:
        unit_vector = 1/unit_vector
        for yshift in range(rad):
            xshift = np.round(unit_vector * yshift)

            k[centre - yshift, int(centre + xshift)] = 1
            k[centre + yshift, int(centre - xshift)] = 1
    
    k /= size

    return(k)


def add_blur(image, imageHeight, kernelSizeToHeight, direction):
  """ Add motion blur to image tensor 
  
  Args:
    image:              tensor, image tensor of shape (?, ?, 3)
    imageHeight:        int, height of image
    kernelSizeToHeight: float, size of kernel wrt image height
    direction:          float, limits of rotation in degrees (-dir, dir)
  """

  #create zero-kernel of size 0.1 of image height
  kernelSize = int( kernelSizeToHeight*imageHeight * random() )
  if kernelSize == 0:
    kernelSize = 1
  
  if kernelSize %2 == 0:
    kernelSize -= 1
  #k = tf.Variable( tf.zeros([kernelSize, kernelSize], tf.float32) )

  d = 2*direction*random() - direction  #[-direction: direction) degrees
  k = createKernel(d, kernelSize)
  k = np.expand_dims(k, axis=2)
  tfkernel = tf.expand_dims(k, 3)
  
  r, g, b = tf.split(image, [1,1,1], 2)
  
  with tf.device('/gpu:0'):
    r = tf.nn.conv2d(
      tf.expand_dims(r,0), 
      tfkernel,
      strides=[1, 1, 1, 1], padding="SAME")
    g = tf.nn.conv2d(
      tf.expand_dims(g,0), 
      tfkernel,
      strides=[1, 1, 1, 1], padding="SAME")
    b = tf.nn.conv2d(
      tf.expand_dims(b,0), 
      tfkernel,
      strides=[1, 1, 1, 1], padding="SAME")
  
  image = tf.concat([r,g,b], 3)
  image = tf.squeeze(image)
  
  return(image)