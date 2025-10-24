import cv2, numpy as np

def detect_text_regions_mser(gray, min_area=60, max_area=10000, delta=5):
    mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area)
    regions, _ = mser.detectRegions(gray)
    boxes = []
    for r in regions:
        x,y,w,h = cv2.boundingRect(r.reshape(-1,1,2))
        ar = w / float(h+1e-6)
        if 0.2 <= ar <= 15.0 and 8 <= h <= 600 and 5 <= w <= 1000:
            boxes.append((x,y,w,h))
    return nms_merge(boxes, iou_thresh=0.3)

def nms_merge(boxes, iou_thresh=0.3):
    if not boxes: return []
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes], dtype=np.float32)
    scores = rects[:,2] - rects[:,0]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(rects[i,0], rects[order[1:],0])
        yy1 = np.maximum(rects[i,1], rects[order[1:],1])
        xx2 = np.minimum(rects[i,2], rects[order[1:],2])
        yy2 = np.minimum(rects[i,3], rects[order[1:],3])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w*h
        area_i = (rects[i,2]-rects[i,0])*(rects[i,3]-rects[i,1])
        area_j = (rects[order[1:],2]-rects[order[1:],0])*(rects[order[1:],3]-rects[order[1:],1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds+1]
    merged = []
    for idx in keep:
        x1,y1,x2,y2 = rects[idx].astype(int)
        merged.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
    return merged
