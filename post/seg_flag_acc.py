import os
from sklearn.metrics import classification_report as sk_clfrpt
from matplotlib import pyplot as plt
import argparse


def recall_report(imgPath, flagsPath, gtPath):
    """ sklearn classification report on flag accuracy 
    
    Args:
        imgPath:    string, path to folder of images (all done seg_post flagging)
        flagsPath:  string, path to csv file of flags
        gtPath:     string, path to csv file of correct flags
    """
    gtCsv = open(gtPath, 'r')                                      #read gt flags csv
    gtFiles = [ os.path.basename(i.split('\n')[0]) for i in gtCsv] #get file names that are manually flagged
    
    flagsCsv = open(flagsPath, 'r')                 #get file names that are flagged (predicted)
    predFiles = [i.split(',')[0] for i in flagsCsv if i.split(',')[1] != ' ']
    
    allImagesNames = os.listdir( os.path.abspath(imgPath) )
    
    clf_pred = [ fn in predFiles for fn in allImagesNames ]   #get 1D-bool arrays of images (flagged = 1)
    gt_pred = [ fn in gtFiles for fn in allImagesNames ]  
    clfrpt_dict = sk_clfrpt(gt_pred, clf_pred, target_names = ['not flagged', 'flagged'])
    
    print(clfrpt_dict)
    return(clfrpt_dict)
    #return(sk_clfrpt['accuracy'])


def alterCorrectToFlagged():
    """ get manually flagged csv from txt that stores file names of correct flagging results """
    
    txtPath = 'C:\\Users\\User\\Desktop\\whizzstuff\\post\\huehuehue1.csv'
    flaggedPath = 'C:\\Users\\User\\Desktop\\whizzstuff\\post\\small_flagged'
    notFlaggedPath = 'C:\\Users\\User\\Desktop\\whizzstuff\\post\\small_post'
    
    f = open(txtPath, 'r')
    outputCsv = [i.split('\n')[0] for i in f]
    f.close()

    notFlaggedImgs = os.listdir( os.path.abspath(notFlaggedPath) )
    notFlaggedImgs = [ os.path.join(notFlaggedPath, i) for i in notFlaggedImgs]

    """
    GET GROUND TRUTH
        - flagged wrongly:      not in outputcsv, exist in flaggedimgs      (don't include)
        - not flagged wrongly:  not in outputcsv, exist in notflaggedimgs   (include)!
        - flagged correctly:    in outputcsv, exist in flaggedimgs          (include)
        - not flagged correctly:  in outputcsv, exist in notflaggedimgs       (don't include)!
    """
    
    for img in notFlaggedImgs:
        if img not in outputCsv:
            outputCsv.append(img)
        else:
            outputCsv.remove(img)
        
    f = open('test.csv', 'w')
    f.write( '\n'.join(outputCsv) )
    f.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-i',
        "--image_path",
        #default= 'C:\\Users\\User\\Desktop\\whizzstuff\\post\\small_seg',
        required=True,
        help= "path to folder with images")

    parser.add_argument(
        '-f1',
        "--flags_path",
        #default='C:\\Users\\User\\Desktop\\whizzstuff\\post\\small_post\\probably_crap.csv',
        required=True,
        help= "path to csv file with predicted flags")
    
    parser.add_argument(
        '-f2',
        "--gt_path",
        #default='C:\\Users\\User\\Desktop\\whizzstuff\\post\\test.csv',
        required=True,
        help= "path to csv file with gt flags")
    
    args = parser.parse_args()
    #alterCorrectToFlagged()
    a = recall_report(args.image_path, args.flags_path, args.gt_path)
    print(a)