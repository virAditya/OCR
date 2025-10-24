import cv2, numpy as np
from .features import hog_feature

def build_font_prototypes(chars=None, sizes=(28,32,36), thickness=(1,2)):
    if chars is None:
        chars = [chr(c) for c in range(ord('0'), ord('9')+1)] + \
                [chr(c) for c in range(ord('A'), ord('Z')+1)] + \
                [chr(c) for c in range(ord('a'), ord('z')+1)]
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX
    ]
    X, Y = [], []
    for ch in chars:
        for f in fonts:
            for s in sizes:
                for th in thickness:
                    canvas = np.full((64,64), 255, dtype=np.uint8)
                    (w, h), base = cv2.getTextSize(ch, f, 1.0, th)
                    scale = max(0.5, s/float(h+1))
                    (w, h), base = cv2.getTextSize(ch, f, scale, th)
                    x = (64 - w)//2
                    y = (64 + h)//2
                    cv2.putText(canvas, ch, (x, y), f, scale, (0,), th, cv2.LINE_AA)
                    fd, _ = hog_feature(canvas)
                    X.append(fd); Y.append(ch)
    X = np.stack(X, axis=0); Y = np.array(Y)
    return X, Y

def knn_predict(X_train, Y_train, x, k=3):
    d = np.linalg.norm(X_train - x[None, :], axis=1)
    idx = np.argsort(d)[:k]
    labels = Y_train[idx]
    eps = 1e-6
    w = 1.0 / (d[idx] + eps)
    best_label, best_score = None, -1
    for c in np.unique(labels):
        score = w[labels == c].sum()
        if score > best_score:
            best_score = score; best_label = c
    conf = float(best_score / (w.sum() + eps))
    return best_label, conf

def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = np.zeros((n+1, m+1), dtype=np.int32)
    dp[:,0] = np.arange(n+1); dp[0,:] = np.arange(m+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return int(dp[n,m])

def correct_with_lexicon(word, lexicon, max_d=2):
    if not lexicon: return word, 1.0
    best, best_d = word, max_d+1
    for w in lexicon:
        d = levenshtein(word, w)
        if d < best_d:
            best_d = d; best = w
            if d == 0: break
    conf = max(0.0, 1.0 - (best_d / float(max_d+1)))
    return best, conf
