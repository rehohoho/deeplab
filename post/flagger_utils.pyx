#cython: language_level=3, cdivision=True
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
ctypedef np.uint8_t DTYPE_UINT8
ctypedef np.int32_t DTYPE_INT32

from collections import deque


cdef struct node:
    short pixelx
    short pixely
    unsigned char label
    int parent
    
cdef class imgGraph:
    cdef node *nodes    #pixel position and label
    cdef float *colmap  #rgb of all labels, float for distance calculation
    cdef float *buckets #number of pixels in each label, float for turning to %
    
    cdef float side, top, bottom, mainRoadLeft, mainRoadRight
    cdef int mainRoadNo, mr_sizeSide, mr_sizeTop
    cdef int nodesNo    #number of pixels
    cdef int colmapNo   #number of labels
    cdef short height
    cdef short width
    #cdef float colorDistBias
    
    
    def __cinit__(self, short height, short width, np.ndarray[np.float32_t, ndim=2] colmap):
        """ initialise pixel number of nodes (xpos, ypos, label) and color map """
        
        cdef int i, j, nodesNo = height*width, colmapLen = 3* len(colmap)
        cdef node *curr
        
        self.width = width
        self.height = height
        self.nodesNo = nodesNo
        self.colmapNo = colmapLen
        
        self.side = 0                       #initiate extreme values for bbox corners
        self.top = 0
        self.bottom = 0
        self.mainRoadNo = 0
        
        self.nodes = <node *> malloc(nodesNo * sizeof(node))
        self.colmap = <float *> malloc(colmapLen * sizeof(float))
        self.buckets = <float *> malloc(colmapLen/3 * sizeof(float))
        
        i = 0
        for h in range(height):
            for w in range(width):
                
                curr = &self.nodes[i]       #init node values: pixelx, pixely, label, parent
                curr.pixelx = w
                curr.pixely = h
                curr.label = -1
                curr.parent = i
                i += 1
                
        for i from 0 <= i < colmapLen by 3: 
            j = i/3
            self.colmap[i] = colmap[j,0]    #init colormap values
            self.colmap[i+1] = colmap[j,1]
            self.colmap[i+2] = colmap[j,2]
            
            self.buckets[j] = 0             #init bucket values
     
     
    def __dealloc__(self):
        """ make memory allocated available again """
        
        free(self.nodes)
        free(self.colmap)
    
    
    cdef int closestCol(self, float r, float g, float b):
        """ find closest label in color map by euclidean distance 
        
        Args:
            pixel: 1D array with 3 channels (RGB)
        Returns:
            closestInd: index of closest color in color map
        """
        
        cdef int i, colmapLen = self.colmapNo, closestInd = -1
        cdef float currDist, rawDist, minDist = 1000000
        
        for i from 0 <= i < colmapLen by 3:
            currDist = ((self.colmap[i] - r)*(self.colmap[i] - r) +
                        (self.colmap[i+1] - g)*(self.colmap[i+1] - g) +
                        (self.colmap[i+2] - b)*(self.colmap[i+2] - b))
            """
            if i == bias1:
                currDist *= self.colorDistBias
            if i == bias2 and bias1 != bias2:
                currDist *= self.colorDistBias
            """
            if currDist < minDist:
                minDist = currDist
                closestInd = i/3
        
        return(closestInd)
    
    
    def load_labels(self, np.ndarray[DTYPE_UINT8, ndim=3] img):
        """ label each pixel by color closest to label in color map, fill buckets
        
        need to calculate closest color due to variation in color caused by:
            - image morphing (to reduce noise)
            - writing and reading image
            
        Args:
            img: 3D array (height, width, RGB channels)
        """
        
        cdef short h, w, height = self.height, width = self.width
        cdef int nodeNo, label
        cdef float r, g, b
        cdef node *n
        
        for h in range(height):
            
            for w in range(width):
                
                nodeNo = h*width + w
                n = &self.nodes[nodeNo]
                
                r = <float>img[h,w,0]
                g = <float>img[h,w,1]
                b = <float>img[h,w,2]
                
                label = self.closestCol( r, g, b )
                self.nodes[nodeNo].label = label
                self.buckets[label] += 1
      
    
    def load_mask(self, np.ndarray[DTYPE_UINT8, ndim=2] mask):

        cdef short h, w, height = self.height, width = self.width
        cdef int n = 0

        for h in range(height):
            
            for w in range(width):

                self.nodes[n].label = mask[h, w]
                n += 1


    def translate_labels(self, transdict):
        cdef int i, nodesNo = self.nodesNo
        
        for i in range(nodesNo):
            self.nodes[i].label = transdict[self.nodes[i].label]
        
    
    cdef int mark_region(self, q, int tar_class):
        """ relabels all pixels connecting (left, right, top, down) with label == tar_class (road)
            
            updates following class variables:
                - buckets
                - road extremes (mainRoadLeft, mainRoadRight)
                - boundary buckets (side, top, bottom)
                
        Args:
            q: deque object, containing index of pixel (ypos*width + xpos)
        """
        
        cdef short height = self.height, width = self.width, x, y, ythres = <short>(0.9*self.height)
        cdef int n
        cdef unsigned char label
        cdef node *nodeRef
        
        while q:                        #loops till empty
            nodeRef = &self.nodes[ q.popleft() ]
            if nodeRef.label == tar_class:
                nodeRef.label = 1
                
                x = nodeRef.pixelx
                y = nodeRef.pixely
                
                self.buckets[1] += 1    #update buckets
                self.buckets[0] -= 1
                
                if x < self.mainRoadLeft and y < ythres:   #record main road extremes
                    self.mainRoadLeft = x
                if x > self.mainRoadRight and y < ythres:
                    self.mainRoadRight = x
                
                n = y*width + x
                if x > 0:               #checks left
                    label = self.nodes[ n-1 ].label
                    if label == tar_class:
                        q.append( n-1 )
                    if label > 1:
                        self.side += 1
                        self.mr_sizeSide += 1
                else:
                    self.side += 1
                    self.mr_sizeSide += 1
                    
                if x < width - 1:       #checks right
                    label = self.nodes[ n+1 ].label
                    if label == tar_class:
                        q.append( n+1 )
                    
                if y > 0:               #checks top
                    label = self.nodes[ n-width ].label
                    if label == tar_class:
                        q.append( n-width )
                    if label > 1:
                        self.top += 1
                        self.mr_sizeTop += 1
                        
                if y < height - 1:      #checks bottom
                    label = self.nodes[ n+width ].label
                    if label == tar_class:
                        q.append( n+width )
                    if label > 1:
                        self.bottom += 1
        
    
    def mark_main_road(self, float noThres):
        """ label pixels (label = 0) connecting from the bottom """
        
        cdef short h, w, height = self.height, width = self.width
        cdef int n, tar_class, mainRoadNo = 0, spaceBet = self.width
        cdef float currWidest = 0, largestLeft, largestRight, mainMinThres
        
        mainMinThres = 0.13 * noThres * <float>self.height

        q = deque()
        
        h = height - 1
        for w in range(width):
            
            n = h*width + w
            tar_class = self.nodes[n].label
            if tar_class == 0:
                q.append(n)
                
                self.mr_sizeSide = 0            #reset metrics for next region
                self.mr_sizeTop = 0
                self.mainRoadLeft = width - 1
                self.mainRoadRight = 0
                
                self.mark_region(q, tar_class)
                
                if self.mainRoadRight - self.mainRoadLeft > currWidest:     #retain widest portion
                    currWidest = self.mainRoadRight - self.mainRoadLeft
                    largestLeft = self.mainRoadLeft
                    largestRight = self.mainRoadRight
                
                if self.mr_sizeSide > mainMinThres and self.mr_sizeTop > mainMinThres and spaceBet >= 0.1*self.width:   #min size and space between for road
                    #print(spaceBet, self.mr_sizeSide, self.mr_sizeTop)
                    mainRoadNo += 1
                    spaceBet = 0
            else:
                spaceBet += 1
        
        self.mainRoadLeft = largestLeft         #retain metrics for flagging
        self.mainRoadRight = largestRight
        self.mainRoadNo = mainRoadNo
    
    
    def show_mask(self):
        """ outputs 2D array with label values for each pixel """
        
        cdef np.ndarray[DTYPE_INT32, ndim=2] mask
        cdef short h,w, height = self.height, width = self.width
        
        mask = np.empty( (height, width), dtype = np.int32 )
        
        for h in range(height):
            
            for w in range(width):
                
                mask[h,w] = self.nodes[h*width + w].label
    
        return(mask)
    
    
    def show_buckets(self):
        """ outputs 1D array with number of pixels in each label """
        
        cdef short i, bucketsLen = self.colmapNo/3
        cdef np.ndarray[np.float32_t, ndim=1] buckets
        
        buckets = np.empty( bucketsLen, dtype = np.float32 )
        
        for i in range(bucketsLen):
            buckets[i] = self.buckets[i]
        
        return(buckets)
    
    """
    Flagging methods tried and failed:
        - size of region
        - road bounding box aspect ratio
        - how much of road bounding box is filled by road
    """
    def show_flags(self, float wrongThres, float noThres, float fatThres):
        
        outputText = ''
        #wrongThres = 0.6, noThres = (300/384) 300, fatThres = 0.9
        
        if self.side + self.side < noThres * <float>self.height:             #min size of road
            outputText = 'too little road'
        else:                                       #ensure self.side is not 0

            if (self.bottom / self.side / 2 > 0.13 and                          #threshold for (num bottom boundaries relative :  num side boundaries)
                (self.side + self.side + self.bottom)/self.top > wrongThres):   #threshold for (num non-bottom : num bottom boundaries)
                    outputText = 'wtf'
            
            elif self.mainRoadRight-self.mainRoadLeft > fatThres*self.width:
                outputText = 'dk where to go'
            
            elif self.mainRoadNo > 1:
                outputText = 'too many main roads'
                
        return( '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s' 
            %(outputText, self.side+self.side, (self.side + self.side + self.bottom)/self.top, self.bottom / self.side / 2,
            self.mainRoadLeft, self.mainRoadRight, self.mainRoadNo, self.side, self.top, self.bottom))
    
    
    def reset_bucketsAndFlags(self):
        """ set all buckets to 0 """
        
        cdef int i, bucketsLen = self.colmapNo/3
        
        for i in range(bucketsLen):
            self.buckets[i] = 0
        
        self.side = 0       #reset bbox corners to extreme values
        self.top = 0
        self.bottom = 0
        self.mainRoadLeft = self.width - 1
        self.mainRoadRight = 0
        self.mainRoadNo = 0

"""
DONE:
    - no road from bottom -> road

    FLAG:
    - too little
    - weird shaped roads
    
    FILL UP SMALL REGIONS:
    - dilation erosion (all regions including road)
    

    SAVE GOOD QUALITY SEG and compare them

NOT DONE:
    - too much road
    - predominantly one class
    
LOOK INTO:
- shape similarity index
- poking at model to get probability of classes (for flagging)

- get all the datasets

"""