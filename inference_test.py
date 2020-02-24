import numpy as np
from PIL import Image

from post.flagger_utils import imgGraph

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


seg_map = np.tile( np.random.choice( (0,1), 256 ), (128, 1) ).astype(np.uint8)
seg_map = np.vstack( (seg_map, np.ones((128, 256), dtype=np.uint8)) )
print(seg_map)

Image.fromarray(
    CITYSCAPES_COLMAP[seg_map].astype(np.uint8)
).show()

road_marker = imgGraph( np.shape(seg_map)[0], np.shape(seg_map)[1], \
                        CITYSCAPES_COLMAP.astype(np.float32))

road_marker.load_mask(seg_map)
road_marker.mark_main_road(0.5)   #threshold to reject road, percentage of height
seg_map = road_marker.show_mask()

seg_image = CITYSCAPES_COLMAP[seg_map].astype(np.uint8)
seg_image = Image.fromarray(seg_image)
seg_image.show()