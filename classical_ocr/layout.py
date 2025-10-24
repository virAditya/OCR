import numpy as np, cv2

def rlsa(binary, h_thresh=25, v_thresh=0):
    img = (binary > 0).astype(np.uint8)
    H, W = img.shape
    out = img.copy()
    if h_thresh > 0:
        for y in range(H):
            row = out[y, :]
            ones = np.where(row == 1)[0]
            if ones.size < 2:
                continue
            for i in range(len(ones)-1):
                if 0 < (ones[i+1] - ones[i] - 1) <= h_thresh:
                    row[ones[i]:ones[i+1]+1] = 1
            out[y, :] = row
    if v_thresh > 0:
        for x in range(W):
            col = out[:, x]
            ones = np.where(col == 1)[0]
            if ones.size < 2:
                continue
            for i in range(len(ones)-1):
                if 0 < (ones[i+1] - ones[i] - 1) <= v_thresh:
                    col[ones[i]:ones[i+1]+1] = 1
            out[:, x] = col
    return (out*255).astype(np.uint8)

def project_valleys(bw, axis=0, min_run=10):
    inv = 255 - bw
    proj = np.sum(inv > 0, axis=axis)
    thresh = np.percentile(proj, 20)
    whitespace = proj <= thresh
    segments, in_blk, start = [], False, 0
    for i, v in enumerate(~whitespace):
        if v and not in_blk:
            in_blk = True; start = i
        elif (not v) and in_blk:
            end = i
            if end - start >= min_run:
                segments.append((start, end))
            in_blk = False
    if in_blk:
        end = len(~whitespace)
        if end - start >= min_run:
            segments.append((start, end))
    return proj, segments

def group_into_words(boxes, line_height_tol=0.5, gap_factor=1.5):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines = []
    for b in boxes:
        x,y,w,h = b
        placed = False
        for line in lines:
            ly = np.median([bb[1] for bb in line])
            lh = np.median([bb[3] for bb in line])
            if abs(y - ly) <= line_height_tol * max(lh,1):
                line.append(b); placed = True; break
        if not placed:
            lines.append([b])
    words = []
    for line in lines:
        line = sorted(line, key=lambda b: b[0])
        avg_w = np.mean([b[2] for b in line]); 
        if np.isnan(avg_w) or avg_w <= 0: avg_w = 15.0
        cur = [line[0]]
        for b in line[1:]:
            prev = cur[-1]
            gap = b[0] - (prev[0] + prev[2])
            if gap > gap_factor * avg_w:
                words.append(cur); cur = [b]
            else:
                cur.append(b)
        if cur: words.append(cur)
    return words
