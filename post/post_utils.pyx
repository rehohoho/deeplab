#cython: language_level=3, cdivision=True
from libc.stdlib cimport malloc, free
from libc.math cimport exp, sqrt, log
from libc.stdlib cimport rand
from libc.string cimport memset

import numpy as np
cimport numpy as np
ctypedef np.uint8_t DTYPE_UINT8
ctypedef np.int32_t DTYPE_INT32
ctypedef np.float32_t DTYPE_FLOAT32

cdef struct edge:
    int node1ind
    int node2ind
    float weight

cdef struct node:
    short pixelx
    short pixely
    unsigned char label

    float prevProb[19]
    float prob[19]
    float confidence
    
    int parent
    float r
    float g
    float b
    int size
    float maxWeight

cdef class imgGraph:
    cdef node *nodes            #pixel position, label, probabilties
    cdef unsigned char *colmap  #rgb of all labels, float for distance calculation
    
    cdef int nodesNo    #number of pixels
    cdef int height
    cdef int width
    
    cdef edge *edges
    cdef int edgesNo    #number of edges

    cdef void generateColMap(self):
        """ Creates a label colormap used in CITYSCAPES segmentation benchmark.

        Returns:
            A Colormap for visualizing segmentation results.
        """

        cdef np.ndarray[np.uint8_t, ndim=2] npcolmap
        cdef int i, j, colmapLen = 19*3
        
        npcolmap = np.array([
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
        ], dtype = np.uint8)

        self.colmap = <unsigned char *> malloc(colmapLen * sizeof(unsigned char))
        cdef unsigned char *label = self.colmap

        for i from 0 <= i < colmapLen by 3: 
            j = i/3
            label[i] = npcolmap[j,0]    #init colormap values
            label[i+1] = npcolmap[j,1]
            label[i+2] = npcolmap[j,2]

    def __cinit__(self, int height, int width):
        """ initialise pixel number of nodes (xpos, ypos, label) and color map 
        
        Args:
            height: int, height of image
            width:  int, width of image
        """
        
        cdef int i, j, h, w, nodesNo = height*width, edgesNo
        cdef node *n
        
        self.width = width
        self.height = height
        self.nodesNo = nodesNo
        
        self.generateColMap()
        self.nodes = <node *> malloc(nodesNo * sizeof(node))
        
        #self.edgesNo = 4*width*height - 3*(width + height) + 2     #include corners
        self.edgesNo = 2*width*height - width - height              #exclude corners
        self.edges = <edge *> malloc(self.edgesNo * sizeof(edge))
        
        i = 0
        for h in range(height):
            for w in range(width):
                
                n = &self.nodes[i]       #init node values: pixelx, pixely, label, parent, confidence
                n.pixelx = w
                n.pixely = h
                n.label = -1
                n.confidence = 1         #no transparency on alpha channel by default
                
                for j in range(19):
                    n.prevProb[j] = 0
                i += 1
     
    def __dealloc__(self):
        """ make memory allocated available again """
        
        free(self.nodes)
        free(self.colmap)
    
    def load_rgb(self, np.ndarray[DTYPE_UINT8, ndim=3] img):
        """ get rgb values of original image into node
            reset values required for graph segmentation
        
        Args:
            img     3D numpy array , dtype = np.float32, shape = (height, width, 3)
        """

        cdef int i, h, w, height = self.height, width = self.width
        cdef node *n

        i = 0
        for h in range(height):
            for w in range(width):
                n = &self.nodes[i]
                n.r = img[h, w, 0]
                n.g = img[h, w, 1]
                n.b = img[h, w, 2]

                n.parent = i
                n.size = 1
                n.maxWeight = 0

                i += 1

    def calc_edges(self):
        """ calculate all adjacent pixel's rgb distance """

        cdef int i, h, w, height = self.height, width = self.width
        cdef int nodeNo, edgeNo, otherNodeNo

        #cdef int alt[12]
        #alt[:] = [-1,1,-width+1,  0,1,1,  1,0,width,  1,1,width+1]     #include corners
        cdef int alt[6]
        alt[:] = [0,1,1,  1,0,width]                                    #exclude corners

        cdef node n1, n2
        cdef edge *e = self.edges
        cdef float rdist, gdist, bdist

        nodeNo = 0
        edgeNo = 0

        for h in range(height):
            for w in range(width):
                
                n1 = self.nodes[nodeNo]

                for i from 0 <= i < 6 by 3:
                    
                    if (0 <= h+alt[i] < height) and (0 <= w+alt[i+1] < width):

                        otherNodeNo = nodeNo + alt[i+2]
                        n2 = self.nodes[otherNodeNo]

                        rdist = n1.r - n2.r
                        gdist = n1.g - n2.g
                        bdist = n1.b - n2.b
                        
                        e = &self.edges[edgeNo]
                        e.node1ind = nodeNo
                        e.node2ind = otherNodeNo
                        e.weight = sqrt(rdist*rdist + gdist*gdist + bdist*bdist)/441.6729559300637
                        edgeNo += 1
                        
                nodeNo += 1
        
        print('edges created %s, total edges no %s' %(edgeNo, self.edgesNo))        

    cdef void sort_edges(self, int start, int end) nogil:
        """ sort edges by weights in ascending order """

        if start >= end: return

        cdef int i = start, j = end
        cdef float piv = self.edges[ rand()%(end-start) + start ].weight

        while i < j:
            while self.edges[i].weight < piv:
                i += 1
            while self.edges[j].weight > piv:
                j -= 1
            if i <= j:
                self.edges[i], self.edges[j] = self.edges[j], self.edges[i]
                i += 1
                j -= 1
        
        self.sort_edges(start, j)
        self.sort_edges(i, end)
    
    def print_edges(self, int num):
        
        self.sort_edges(0, self.edgesNo)
        
        cdef int i

        print('first bunch: \n')
        for i in range(num):
            print( self.edges[i].node1ind, self.edges[i].node2ind, self.edges[i].weight, '\n')
        print('last bunch: \n')
        for i from self.edgesNo - 1 >= i > self.edgesNo -1-num by 1:
            print( self.edges[i].node1ind, self.edges[i].node2ind, self.edges[i].weight, '\n')
    
    cdef int findParent(self, int n) nogil:     #path compression for disjoint set
        if self.nodes[n].parent != n:
            self.nodes[n].parent = self.findParent( self.nodes[n].parent )
        return(self.nodes[n].parent)

    def oversegment(self, float thres):
        """ graph segmentation
        adds class probabilities of pixels in region
        label region highest probability 

        finds boundaries by euclidean rgb distance and tries to enforce it
        """

        cdef int i, j, edgesNo = self.edgesNo, nodesNo = self.nodesNo
        cdef int root1, root2, regionlabel
        cdef float edgeWeight, maxProb
        cdef edge e
        #cdef node *n1
        #cdef node *n2

        self.sort_edges(0, edgesNo)
        print('edges sorted')

        for i in range(edgesNo):

            e = self.edges[i]
            root1 = self.findParent( e.node1ind )
            root2 = self.findParent( e.node2ind )
            
            if root1 != root2:
                
                #n1 = &self.nodes[root1]
                #n2 = &self.nodes[root2]
                edgeWeight = e.weight

                if (edgeWeight < (self.nodes[root1].maxWeight + thres/self.nodes[root1].size) and
                    edgeWeight < (self.nodes[root2].maxWeight + thres/self.nodes[root2].size)):
                    self.nodes[root2].parent = self.nodes[root1].parent
                    self.nodes[root1].maxWeight = edgeWeight
                    self.nodes[root1].size += self.nodes[root2].size
                    for j in range(19):
                        self.nodes[root1].prob[j] += self.nodes[root2].prob[j]
        
        for i in range(nodesNo):

            #n1 = &self.nodes[ self.nodes[i].parent ]
            maxProb = 0
            
            for j in range(19):
                if self.nodes[ self.nodes[i].parent ].prob[j] > maxProb:
                    maxProb = self.nodes[ self.nodes[i].parent ].prob[j]
                    regionlabel = j
            
            self.nodes[i].label = regionlabel

    def load_logits(self, np.ndarray[DTYPE_FLOAT32, ndim=2] prob):
        """ from 2D array of logits, load probabilities for each pixel 
        
        Args:
            prob:   2D numpy array, dtype = np.float32, shape = (pixelNo, 19)
                    probabilities of each class for each pixel from left top to bottom right
        """

        cdef int i, j , height = self.height, width = self.width, nodesNo = self.nodesNo
        cdef node *n

        assert prob.size == height*width*19, 'Prob array size is %s, expected %s' %(prob.size, height*width*19)

        i = 0
        for i in range(nodesNo):
            
            n = &self.nodes[i]

            for j in range(19):
                n.prob[j] = prob[i, j]
    
    def softmax_logits(self):
        """ softmax logits of all pixels, can only extract logits from frozen graph """

        cdef int i, j, nodesNo = self.nodesNo
        cdef float expsum, expcurr
        cdef node *n

        for i in range(nodesNo):
            
            n = &self.nodes[i]
            expsum = 0

            for j in range(19):
                expcurr = exp(n.prob[j])
                n.prob[j] = expcurr
                expsum += expcurr
            
            for j in range(19):
                n.prob[j] /= expsum
    
    def pushback_logits(self):
        """ shift current frame prob to previous frame prob """

        cdef int i, j, nodesNo = self.nodesNo
        cdef node *n

        for i in range(nodesNo):
            n = &self.nodes[i]    
            for j in range(19):
                n.prevProb[j] = n.prob[j]

    def calc_confidence(self, float thres):
        """ updates confidence level for each pixel 
        
        FAILED BS:
        confidence = area between highest prob and other class probabilities
            - favour sharp peak
        
        confidence = 1 / sum(candidates)
        where candidates have prob = thes * max_prob
        """
        
        cdef int i, j, nodesNo = self.nodesNo
        cdef float maxProb, conf
        cdef node *n

        for i in range(nodesNo):
            
            n = &self.nodes[i]
            
            maxProb = 0
            for j in range(19):
                if n.prob[j] > maxProb:
                    maxProb = n.prob[j]
            
            conf = 0
            for j in range(19):
                if n.prob[j] / maxProb < thres:
                    n.prob[j] = 0
                else:
                    conf += n.prob[j]
            
            n.confidence = maxProb / conf
    
    def post(self, float mul):
        """ use previous frame's class to influence labels"""

        cdef int i, j, nodesNo = self.nodesNo
        cdef unsigned char label
        cdef float maxProb
        cdef node *n

        for i in range(nodesNo):
            
            n = &self.nodes[i]
            maxProb = 0

            for j in range(19):
                
                n.prevProb[j] += n.prob[j] * mul * n.confidence
                
                if n.prevProb[j] > maxProb:
                    maxProb = n.prevProb[j]
                    label = j
                
                n.label = label

    def no_post(self):
        """ choose highest probability as label """

        cdef int i, j, nodesNo = self.nodesNo
        cdef unsigned char label
        cdef float maxProb
        cdef node *n

        for i in range(nodesNo):
            
            n = &self.nodes[i]
            maxProb = 0

            for j in range(19):
                
                if n.prob[j] > maxProb:
                    maxProb = n.prob[j]
                    label = j
                
                n.label = label
    
    def show_confidence(self):
        """ outputs 2D array with label values for each pixel """
        
        cdef np.ndarray[DTYPE_FLOAT32, ndim=2] conf
        cdef int h,w, height = self.height, width = self.width
        cdef int pixelsSoFar

        conf = np.empty( (height, width), dtype = np.float32 )
        
        pixelsSoFar = 0

        for h in range(height):
            for w in range(width):
                conf[h,w] = self.nodes[pixelsSoFar + w].confidence

            pixelsSoFar += width

        return(conf)

    def show_mask(self):
        """ outputs 2D array with label values for each pixel """
        
        cdef np.ndarray[DTYPE_UINT8, ndim=2] mask
        cdef int h,w, height = self.height, width = self.width
        cdef int pixelsSoFar

        mask = np.empty( (height, width), dtype = np.uint8 )
        
        pixelsSoFar = 0

        for h in range(height):
            for w in range(width):
                mask[h,w] = self.nodes[pixelsSoFar + w].label

            pixelsSoFar += width

        return(mask)
    
    def show_image(self, bint conf):
        """ outputs 3D image array according to pixel label """
        
        cdef np.ndarray[DTYPE_UINT8, ndim=3] img
        cdef unsigned char *col = self.colmap
        cdef int h,w, height = self.height, width = self.width
        cdef int pixelsSoFar, pixelLabel, colInd
        cdef node *n

        img = np.empty( (height, width, 4), dtype = np.uint8 )

        pixelsSoFar = 0

        for h in range(height):
            for w in range(width):
                
                n = &self.nodes[pixelsSoFar + w]
                pixelLabel = <int>n.label
                colInd = pixelLabel * 3
                img[h,w,0] = col[colInd]
                img[h,w,1] = col[colInd+1]
                img[h,w,2] = col[colInd+2]
                img[h,w,3] = <int>(conf * (255 * n.confidence))

            pixelsSoFar += width
        
        return(img)

    def show_parents(self):
        """ outputs 2D array with parent values for each pixel """
        
        cdef np.ndarray[DTYPE_INT32, ndim=2] mask
        cdef int h,w, height = self.height, width = self.width
        cdef int n

        mask = np.empty( (height, width), dtype = np.int32 )
        
        n = 0
        
        for h in range(height):
            for w in range(width):
                mask[h,w] = self.nodes[n].parent
                n += 1
        
        print(mask)
        
        return(mask)


def softmax_logits_forcrf(int height, int width, np.ndarray[DTYPE_FLOAT32, ndim=3] logits):
    """ softmax logits of all pixels, can only extract logits from frozen graph """

    assert np.shape(logits) == (height, width, 19)
    
    cdef np.ndarray[DTYPE_FLOAT32, ndim=2] nll
    cdef int h, w, p, row
    cdef float expsum, expcurr

    nll = np.empty( (19, height*width), dtype=np.float32 )
    
    for h in range(height):
        
        row = h*width

        for w in range(width):

            expsum = 0
            for p in range(19):
                expcurr = exp(logits[h, w, p])
                expsum += expcurr
                logits[h, w, p] = expcurr
            
            for p in range(19):
                nll[p, row + w] = logits[h, w, p] / expsum
    
    return(nll)


def argmax_logits(int height, int width, np.ndarray[DTYPE_FLOAT32, ndim=3] logits):
    """ get segmentation map by highest probability """
    
    assert np.shape(logits) == (height, width, 19)

    cdef np.ndarray[DTYPE_INT32, ndim=2] mask
    cdef int p, label
    cdef float maxProb
    
    mask = np.empty( (height, width), dtype = np.int32)

    for h in range(height):
        for w in range(width):
            
            maxProb = 0
            for p in range(19):
                if logits[h, w, p] > maxProb:
                    maxProb = logits[h, w, p]
                    label = p

            mask[h, w] = label
    
    return(mask)


def adhere_boundary(int height, int width, int n_segments, np.ndarray[DTYPE_INT32, ndim=2] im_superPix, np.ndarray[DTYPE_INT32, ndim=2] mask):
    """ alter labels to adhere to boundaries from superpixels """
    
    return


def calc_confidence_forcrf(float thres, float minProb, int height, int width, np.ndarray[DTYPE_FLOAT32, ndim=3] logits, np.ndarray[DTYPE_UINT8, ndim=3] img):
    """ softmax logits for crf model
        calculate confidence metric
        calculate metrics to tune crf model

    Args:
        thres           float, max confidence for pixel to be counted into n_notconf
        minProb         float, max percentage of maxProb for class probability to be accounted for in confidence metric
        height          int, height for image and logits
        width           int, width for image and logits
        logits          3D numpy.array (height, width, 19), outputs of model for each class, not softmaxed
        img             3D numpy.array (height. width, 3), RGB channels of image

    Returns:
        nll             2D numpy.array (19, pixels), softmaxed logits for crf model
        conf            2D numpy.array (height, width), confidence [0-1] for each pixel
        n_notconf       int number of pixels with confidence below thres
        colorfulness    metric from Hassler and Susstrunk
    """
    
    assert np.shape(logits) == (height, width, 19)
    assert np.shape(img) == (height, width, 3)

    cdef np.ndarray[DTYPE_FLOAT32, ndim=2] nll
    cdef np.ndarray[DTYPE_FLOAT32, ndim=2] conf
    cdef int h, w, p, n_notconf = 0, row = 0
    cdef float expsum, expcurr, maxProb, pixelConf
    nll = np.empty( (19, height*width), dtype=np.float32 )
    conf = np.empty( (height, width), dtype=np.float32 )

    cdef int colorfulnessInd = 0
    cdef float *rg
    cdef float *yb
    cdef float currrg, curryb, rgmean=0, rgsd=0, ybmean=0, ydsd=0, colorfulness
    rg = <float *> malloc ( height*width * sizeof (float) )
    yb = <float *> malloc ( height*width * sizeof (float) )
    
    for h in range(height):

        for w in range(width):
            
            expsum = 0
            for p in range(19):
                expcurr = exp(logits[h, w, p])
                expsum += expcurr
                logits[h, w, p] = expcurr
            
            maxProb = 0                     #find max probability
            for p in range(19):
                logits[h, w, p] /= expsum   #apply softmax
                nll[p, row + w] = logits[h, w, p]
                if logits[h, w, p] > maxProb:
                    maxProb = logits[h, w, p]

            pixelConf = 0                   #sum probabilities > 0.1 max
            for p in range(19):
                if logits[h, w, p] / maxProb > minProb:
                    pixelConf += logits[h, w, p]
            
            pixelConf = maxProb / pixelConf #conf = max / (sum of probs > 0.1max), greater uncertainty lower conf
            conf[h, w] = pixelConf

            if pixelConf < thres:           #count number of pixels not confident
                n_notconf += 1
                
                # track colour difference
                currrg = img[h, w, 0] - img[h, w, 1]
                curryb = 0.5 * (img[h, w, 0] + img[h, w, 1]) - img[h, w, 2]
                
                rg[colorfulnessInd] = currrg
                yb[colorfulnessInd] = curryb
                
                rgmean += currrg
                ybmean += curryb
                colorfulnessInd += 1

        row += width

    # calculate colorfulness of not confidence portion
    rgmean /= colorfulnessInd
    ybmean /= colorfulnessInd

    for p in range(colorfulnessInd):
        currrg = rg[p] - rgmean
        rgsd += currrg * currrg

        curryb = yb[p] - ybmean
        ybsd = curryb * curryb
    
    rgsd /= colorfulnessInd     # variance of rg and yb distances
    ybsd /= colorfulnessInd
    colorfulness = sqrt(rgsd + ybsd) + 0.3 * sqrt( rgmean*rgmean + ybmean*ybmean)

    free(rg)
    free(yb)

    return(nll, conf, n_notconf, colorfulness)


def colorfulness_metric(int height, int width, np.ndarray[DTYPE_UINT8, ndim=3] img):
    """ calculate colorfulness of image (Hassler and Susstrunk) """

    cdef int h, w, p, pixelNo
    cdef float *rg
    cdef float *yb
    cdef float currrg, curryb, rgmean=0, rgsd=0, ybmean=0, ydsd=0

    pixelNo = height * width
    rg = <float *> malloc ( pixelNo * sizeof (float) )
    yb = <float *> malloc ( pixelNo * sizeof (float) )

    p = 0
    for h in range(height):     # rg and yb distances
        for w in range(width):
            
            currrg = img[h, w, 0] - img[h, w, 1]
            curryb = 0.5 * (img[h, w, 0] + img[h, w, 1]) - img[h, w, 2]
            
            rg[p] = currrg
            yb[p] = curryb
            
            rgmean += currrg
            ybmean += curryb

            p += 1
    
    rgmean /= p                 # mean of rg and yb distances
    ybmean /= p

    for p in range(pixelNo):
        currrg = rg[p] - rgmean
        rgsd += currrg * currrg

        curryb = yb[p] - ybmean
        ybsd = curryb * curryb
    
    rgsd /= p                   # variance of rg and yb distances
    ybsd /= p

    free(rg)
    free(yb)

    return( sqrt(rgsd + ybsd) + 0.3 * sqrt( rgmean*rgmean + ybmean*ybmean) )


def color_histogram_forcrf(int height, int width, np.ndarray[DTYPE_UINT8, ndim=3] img):
    """ calculate color histogram for full image 
    
    Args:
        height      int, height for image and logits
        width       int, width for image and logits
        img         3D numpy.array (height, width, 3), RGB channel of image
    
    Returns:
        colhist_np      1D np.array (60), 20 buckets for RGB respectively
    """

    assert np.shape(img) == (height, width, 3)

    cdef int h, w
    cdef int *colhist
    cdef np.ndarray[DTYPE_INT32, ndim=1] colhist_np
    
    colhist = <int *> malloc (60 * sizeof(int))
    memset( colhist, 0, 60*sizeof(int) )
    colhist_np = np.empty( (60), dtype=np.int32 )
    
    for h in range(height):
        
        for w in range(width):
            
            colhist[ img[h, w, 0]/20 ] += 1
            colhist[ 20+ img[h, w, 1]/20 ] += 1
            colhist[ 40+ img[h, w, 2]/20 ] += 1
    
    for h in range(60):
        colhist_np[h] = colhist[h]

    free(colhist)
    return(colhist_np)
                
"""
IDEAS:
- use previous frame to help determine class
    - when model is unsure of pixel's class (how close the prob)
        thres: c * highest_prob[x]
    - use previous frame's probabilties to help (how much)
        new prob: k * class_prob[x-1] + class_prob[x]

    - use surrounding pixels to help determine class

    - conventional methods to help locate boundaries (pictures v diff from training samples)
        - check if confidence level << average by thres
        - graph segmentation
        - label class that has to majority
    
    - determine crf thresholds:
        - col threshold:
            - color histogram of image
            - color histogram of not confidence portions
        - pos threshold:
            - number of pixels not confident
"""