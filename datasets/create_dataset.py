
import argparse
import os
from shutil import copy
from PIL import Image
import numpy as np

from label_handler.label_handler import *
from datetime import datetime


def create_directories(newdataset_path, imagefolder_path, segfolder_path):
    """ create lists of directories to process images """

    segPath = os.path.join(newdataset_path, 'seg')
    paths = [newdataset_path, segPath,
            os.path.join(newdataset_path, 'train'),
            os.path.join(newdataset_path, 'val'),
            os.path.join(newdataset_path, 'test')
            ]
    imgPaths = [os.path.join(imagefolder_path, 'train'),#image path has to contain train, val, test
                os.path.join(imagefolder_path, 'val'), 
                os.path.join(imagefolder_path, 'test')]
    segPaths = [os.path.join(segfolder_path, 'train'),  #seg path has to contain train, val
                os.path.join(segfolder_path, 'val'),
                None]
    
    for pathname in paths:
        if not os.path.exists(pathname):
            os.mkdir(pathname)
            print('created directory %s' %pathname)
    
    return(segPath, paths, imgPaths, segPaths)


def create_trainvaltest(dataset_source,
    imagefolder_path, image_name_format, requiredType,
    segfolder_path, seg_name_format,
    dataset_path, datasetName,
    trainNo, valNo, testNo):
    """ Process images fromraw dataset into directories compatible with build_dataset 
    
    Args:
        dataset_source      string, type of dataset used (BDD, CITYSCAPES or MAPILLARY only)
        imagefolder_path    path,   dataset images folder
        image_name_format   string, image naming convention, eg. .jpg (MAPILLARY) or _leftImg8bit.png (CITYSCAPES)
        segfolder_path      path,   dataset labels folder
        seg_name_format     string, label naming convention, eg. .png (MAPILLARY) or _train_id.png (BDD)
        dataset_path        path,   deeplab dataset folder
        datasetName         string, desired folder name
        trainNo             int,    train split
        valNo               int,    val split
        testNo              int,    test split
    """

    numOfImages = [trainNo, valNo, testNo]
    numOfImages = [int(i) for i in numOfImages]

    newdataset_path = os.path.join( dataset_path, datasetName)
    segPath, paths, imgPaths, segPaths = create_directories(new_dataset_path, imagefolder_path, segfolder_path)
    
    imgType = image_name_format[-3:] == requiredType    #enforce image type
    cityscapes = dataset_source=="CITYSCAPES"           #check if need convert labels
    mapillary = dataset_source=="MAPILLARY"             #check if need convert labels

    label_converter = imgHandler()

    for directoryNo in range(3):
        
        currImgPath = imgPaths[directoryNo]
        currPath = paths[directoryNo+2]
        currSegPath = segPaths[directoryNo]

        currMaxNo = numOfImages[directoryNo]
        i = 0

        walk = os.walk(currImgPath)

        for path, directories, files in walk:
            if files == []:
                continue
            else:
                for name in files:
                    
                    f = os.path.join(path,name)

                    if not imgType:
                        im = Image.open(f)
                        baseName = '%s.%s' %( os.path.splitext( os.path.basename(f) )[0], requiredType)
                        im.save(os.path.join( currPath, baseName ) )
                    elif mapillary:
                        im = Image.open(f)
                        im = im.resize((2048,1024), Image.ANTIALIAS)
                        baseName = '%s.%s' %( os.path.splitext( os.path.basename(f) )[0], requiredType)
                        im.save(os.path.join( currPath, baseName ) )
                    else:
                        copy(f,currPath)
                    
                    if currSegPath != None:
                        
                        s = f.replace( currImgPath, currSegPath )
                        s = s.replace( image_name_format, seg_name_format )
                        
                        if cityscapes:
                            label_converter.label_cityscapes(s, segPath)
                        elif mapillary:
                            label_converter.label_mapillary(s, segPath)
                        else:
                            copy(s,segPath)

                        oldname = os.path.join( segPath, os.path.basename(s) )
                        newname = oldname.replace( seg_name_format, image_name_format[:-4]+'_label.png' )
                        os.rename( oldname, newname )

                    i += 1
                    if i >= currMaxNo:
                        break
                if i >= currMaxNo:
                    break
                    

def create_index(dataset_path, datasetName):
    """ Generate text file containing paths of data splits for build_dataset
    
    Args:
        dataset_path    path    deeplab datasets folder
        datasetName     string  desired name for folder in dataset_path
    """

    newdataset_path = os.path.join( args.dataset_path, datasetName)
    trainPath = os.path.join(newdataset_path, 'train')
    valPath = os.path.join(newdataset_path, 'val')
    testPath = os.path.join(newdataset_path, 'test')
    indexPath = os.path.join(newdataset_path, 'index')

    if not os.path.exists(indexPath):
        os.mkdir( indexPath )
        
        print('created index directory')
    
    imagePaths = [trainPath, valPath]
    indexFileNames = ['train.txt', 'val.txt']
    
    for directoryNo in range(2):
        fileNames = os.listdir(imagePaths[directoryNo])
        outputFileName = os.path.join(indexPath, indexFileNames[directoryNo])
        outputFile = open( outputFileName, 'a' )
        
        tempstr = []
        if os.stat(outputFileName).st_size != 0:
            tempstr.append('\n')

        for name in fileNames:
            tempstr.append('%s\n' %os.path.splitext(name)[0] )
        tempstr[-1] = tempstr[-1].strip('\n')

        outputFile.write( ''.join(tempstr) )
        outputFile.close()


def create_imagesfolder(dataset_path, datasetName):
    """ Combine directories for train/val 
    
    Args:
        dataset_path    path    deeplab datasets folder
        datasetName     string  desired name for folder in dataset_path
    """

    newdataset_path = os.path.join( args.dataset_path, datasetName)
    trainPath = os.path.join(newdataset_path, 'train')
    valPath = os.path.join(newdataset_path, 'val')
    imgPath = os.path.join(newdataset_path, 'images')
    
    paths = [trainPath, valPath]

    if not os.path.exists(imgPath):
        os.mkdir( imgPath )

        print('Created images folder')

    for path in paths:
        filenames = os.listdir( path )
        for filename in filenames:
            newPath = os.path.join( imgPath, filename )
            os.rename( os.path.join(path,filename), newPath )

        os.rmdir(path)


if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-src',
        "--dataset_source",
        required=True,
        help= "source of dataset")

    parser.add_argument(
        '-i',
        "--imagefolder_path",
        required=True,
        help= "path to folder with images")

    parser.add_argument(
        '-fi',
        "--image_name_format",
        required=True,
        help= "name to append when searching images in directory")

    parser.add_argument(
        '-s',
        "--segfolder_path",
        required=True,
        help= "path to folder with segmentations")

    parser.add_argument(
        '-fs',
        "--seg_name_format",
        required=True,
        help= "name to append when searching seg in directory")

    parser.add_argument(
        '-o',
        "--dataset_path",
        required=True,
        help= "path to deeplab/datasets")
    
    parser.add_argument(
        '-n',
        "--dataset_name",
        required=True,
        help= "name of data set")
    
    parser.add_argument(
        '-fo',
        "--output_format",
        required=True,
        help= "image type for training")
    
    parser.add_argument(
        '-t1',
        "--train",
        required=True,
        help= "number of training images")
    
    parser.add_argument(
        '-t2',
        "--val",
        required=True,
        help= "number of validation images")
    
    parser.add_argument(
        '-t3',
        "--test",
        required=True,
        help= "number of test images")
    
    args = parser.parse_args()

    print('Appending data from %s and %s' %(args.imagefolder_path, args.segfolder_path))
    
    print(datetime.now().time())
    create_trainvaltest(
        args.dataset_source,
        args.imagefolder_path, args.image_name_format, args.output_format,
        args.segfolder_path, args.seg_name_format,
        args.dataset_path, args.dataset_name, 
        args.train, args.val, args.test
        )
    print(datetime.now().time())
    create_index(args.dataset_path, args.dataset_name)
    create_imagesfolder(args.dataset_path, args.dataset_name)

"""
# run on dataset_name/seg to check labels are valid

import os
import numpy as np
from PIL import Image
filenames = os.listdir()
for filename in filenames:
    im = np.unique( np.array( Image.open(filename) ) )
    for label in im:
            if not 0 <= label < 19 and not label == 255:
                    print(filename, label)
"""