import cv2, math, numpy as np

def rotate_image(image, angle_deg, border_value=255):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

def cc_boxes(bw, min_w=3, min_h=8, max_w=2000, max_h=2000):
    inv = 255 - bw  # text as white
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    boxes = []
    for i in range(1, n):
        x,y,w,h,a = stats[i]
        if min_w <= w <= max_w and min_h <= h <= max_h:
            boxes.append((x,y,w,h))
    return boxes

def crop_pad(img, box, pad=2, min_size=4):
    H, W = img.shape[:2]
    x,y,w,h = box
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    c = img[y0:y1, x0:x1]
    if c.shape[0] < min_size or c.shape[1] < min_size:
        return None
    return c
