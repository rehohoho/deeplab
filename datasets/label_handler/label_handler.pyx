#cython: language_level=3, cdivision=True
from PIL import Image
import os

import numpy as np
cimport numpy as np
ctypedef np.uint8_t DTYPE_UINT8

from libc.stdlib cimport malloc, free


cdef class imgHandler:

    cdef int *city_train_label
    cdef int *mapillary_label

    cdef void setupCityScapes(self):
        self.city_train_label = <int *> malloc ( 33 * sizeof(int))
        cdef int *label_arr = self.city_train_label

        label_arr[0] = 255
        label_arr[1] = 255
        label_arr[2] = 255
        label_arr[3] = 255
        label_arr[4] = 255
        label_arr[5] = 255
        label_arr[6] = 255
        label_arr[7] = 0
        label_arr[8] = 1
        label_arr[9] = 255
        label_arr[10] = 255
        label_arr[11] = 2
        label_arr[12] = 3
        label_arr[13] = 4
        label_arr[14] = 255
        label_arr[15] = 255
        label_arr[16] = 255
        label_arr[17] = 5
        label_arr[18] = 255
        label_arr[19] = 6
        label_arr[20] = 7
        label_arr[21] = 8
        label_arr[22] = 9
        label_arr[23] = 10
        label_arr[24] = 11
        label_arr[25] = 12
        label_arr[26] = 13
        label_arr[27] = 14
        label_arr[28] = 15
        label_arr[29] = 255
        label_arr[30] = 255
        label_arr[31] = 16
        label_arr[32] = 17
        label_arr[33] = 18

    #if class does not exist in cityscapes, it is not trained on
    cdef void setupMapillary(self):

        self.mapillary_label = <int *> malloc ( 66 * sizeof(int))
        cdef int *label_arr = self.mapillary_label

        label_arr[0] = 255      #bird
        label_arr[1] = 255      #animal
        label_arr[2] = 255      #curb
        label_arr[3] = 4    #fence
        label_arr[4] = 255      #guardrail
        label_arr[5] = 255      #other barriers
        label_arr[6] = 3    #wall
        label_arr[7] = 0    #bike lane (road)
        label_arr[8] = 0    #crosswalk (road)
        label_arr[9] = 0    #curb cut (road)
        label_arr[10] = 0   #parking (road)
        label_arr[11] = 1   #pedestrain area
        label_arr[12] = 255     #rail track
        label_arr[13] = 0   #road
        label_arr[14] = 0   #service lane (road)
        label_arr[15] = 1   #sidewalk
        label_arr[16] = 255     #bridge
        label_arr[17] = 2   #building
        label_arr[18] = 255     #tunnel
        label_arr[19] = 11  #person
        label_arr[20] = 12  #cyclist (rider)
        label_arr[21] = 12  #motorcyclist (rider)
        label_arr[22] = 12  #other rider
        label_arr[23] = 255     #lane marking -crosswalk
        label_arr[24] = 255     #lane marking - general
        label_arr[25] = 9   #mountain (terrain)
        label_arr[26] = 9   #sand (terrain)
        label_arr[27] = 10  #sky
        label_arr[28] = 9   #snow (terrain)
        label_arr[29] = 9   #terrain
        label_arr[30] = 8   #vegetation
        label_arr[31] = 9   #water (terrain)
        label_arr[32] = 255     #banner
        label_arr[33] = 255     #bench
        label_arr[34] = 255     #bike rack
        label_arr[35] = 255     #billboard
        label_arr[36] = 255     #catch-basin
        label_arr[37] = 255     #cctv camera
        label_arr[38] = 255     #fire hydrant
        label_arr[39] = 255     #junction box
        label_arr[40] = 255     #mailbox
        label_arr[41] = 255     #manhole
        label_arr[42] = 255     #phone booth
        label_arr[43] = 255     #pothole
        label_arr[44] = 5   #street light (pole)
        label_arr[45] = 5   #pole
        label_arr[46] = 7   #traffic sign frame (traffic sign)
        label_arr[47] = 5   #utility pole (pole)
        label_arr[48] = 6   #traffic light
        label_arr[49] = 7   #traffic sign back
        label_arr[50] = 7   #traffic sign front
        label_arr[51] = 255     #trash can
        label_arr[52] = 18  #bicycle
        label_arr[53] = 255     #boat
        label_arr[54] = 15  #bus
        label_arr[55] = 13  #car
        label_arr[56] = 255     #caravan
        label_arr[57] = 17  #motorcycle
        label_arr[58] = 255     #on rails
        label_arr[59] = 255     #other vehicle
        label_arr[60] = 255     #trailer
        label_arr[61] = 14  #truck
        label_arr[62] = 255     #wheeled slow
        label_arr[63] = 255     #car mount
        label_arr[64] = 255     #ego vehicle
        label_arr[65] = 255 #unlabelled

    def __cinit__(self):
        
        self.setupCityScapes()
        self.setupMapillary()
    
    def __dealloc__(self):
        """ make memory allocated available again """
        
        free(self.city_train_label)
        free(self.mapillary_label)
    
    def label_cityscapes(self, imgPath, segPath):
        
        cdef np.ndarray[DTYPE_UINT8, ndim=2] np_im
        cdef int h, w, height, width

        im = Image.open(imgPath)
        np_im = np.array(im)
        height, width = np.shape(np_im)

        for h in range(height):
            for w in range(width):
                
                np_im[h, w] = self.city_train_label[ np_im[h, w] ]
        
        im = Image.fromarray(np_im)
        return(im)
    
    def label_mapillary(self, imgPath, segPath):
        
        cdef np.ndarray[DTYPE_UINT8, ndim=2] np_im
        cdef int h, w, height, width

        im = Image.open(imgPath)
        np_im = np.array(im)
        height, width = np.shape(np_im)

        for h in range(height):
            for w in range(width):
                
                np_im[h, w] = self.mapillary_label[ np_im[h, w] ]
        
        im = Image.fromarray(np_im)
        im = im.resize((2048,1024))
        return(im)