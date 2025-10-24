import cv2, math, numpy as np
from skimage.filters import threshold_sauvola
from .utils import rotate_image

def to_gray_denoise(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return gray

def estimate_skew_correct(gray):
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=120, maxLineGap=10)
    angle_deg = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for l in lines[:2000]:
            x1,y1,x2,y2 = l[0]
            ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            if -45 <= ang <= 45:
                angles.append(ang)
        if len(angles) > 0:
            angle_deg = float(np.median(angles))
    corrected = rotate_image(gray, -angle_deg, border_value=255)
    return angle_deg, edges, lines, corrected

def otsu_binarize(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw

def sauvola_binarize(gray, window=25, k=0.34):
    th = threshold_sauvola(gray, window_size=window, k=k)
    bw = (gray > th).astype(np.uint8)*255
    return bw
