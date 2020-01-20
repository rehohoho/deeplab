
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import sys
import build_data
from six.moves import range
import tensorflow as tf

import argparse
from os import environ

FLAGS = tf.app.flags.FLAGS


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  max_height = 0
  max_width = 0

  for shard_id in range(_NUM_SHARDS):
    
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      
      for i in range(start_idx, end_idx):
        
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        image_data = tf.gfile.GFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + FLAGS.label_format)
        seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        
        if height > max_height:
          max_height = height
        if width > max_width:
          max_width = width
        
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    
    sys.stdout.write('\n')
    sys.stdout.flush()
  
  print('max height: %s, max width %s' %(max_height, max_width))


def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()

  parser.add_argument(
        '-i',
        "--image_folder",
        required=True,
        help= "path to folder with images")

  parser.add_argument(
        '-fi',
        "--image_format",
        required=True,
        help= "name to append when searching images in directory")

  parser.add_argument(
        '-s',
        "--semantic_segmentation_folder",
        required=True,
        help= "path to folder with segmentations")

  parser.add_argument(
        '-fs',
        "--label_format",
        required=True,
        help= "name to append when searching seg in directory")

  parser.add_argument(
        '-l',
        "--list_folder",
        required=True,
        help= "path to folder with txtfiles containing image names")

  parser.add_argument(
        '-o',
        "--output_dir",
        required=True,
        help= "path to folder to output tfrecords")
  
  parser.add_argument(
        '-gpu',
        "--gpu_utilise",
        required=True,
        help= "use GPU or not")
  
  FLAGS, unparsed = parser.parse_known_args()
  
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
    print('created tfrecord directory')

  if FLAGS.gpu_utilise == "-1":
    environ["CUDA_VISIBLE_DEVICES"] = "-1" #disables GPU
  
  _NUM_SHARDS = 4

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
