import cv2, numpy as np
from skimage.feature import hog
from skimage.exposure import rescale_intensity

def hog_feature(image_gray, size=(64,64)):
    g = cv2.resize(image_gray, size, interpolation=cv2.INTER_AREA)
    fd, viz = hog(g, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
                  visualize=True, feature_vector=True)
    viz = rescale_intensity(viz, in_range=(0, 10), out_range=(0,255)).astype(np.uint8)
    return fd.astype(np.float32), viz
