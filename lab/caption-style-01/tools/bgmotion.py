"""Background similarity motion between consecutive frames (ORB + RANSAC, text regions masked out)."""
import numpy as np, cv2, json, sys
from common import frames, W, H
a = frames()
CACHE = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
reg = json.load(open(f'{CACHE}/registration.json'))
CUTS = [1, 9, 11, 17, 23, 37, 41, 50, 53, 70, 86, 100, 106, 111, 122, 132, 137, 148]
def text_boxes(n):
    boxes = []
    for wid, r in reg.items():
        for fr in r['frames']:
            if fr['frame'] == n and fr['score'] > 0.6:
                boxes.append(fr['bbox'])
    return boxes
def gray(n):
    f = np.asarray(a[n-1])
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    # local contrast normalisation helps dark shots
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cl.apply(g)
def mask_for(n, m):
    M = np.full((H, W), 255, np.uint8)
    for b in text_boxes(n) + text_boxes(m):
        x0, y0, x1, y1 = [int(v) for v in b]
        cv2.rectangle(M, (x0-25, y0-25), (x1+25, y1+25), 0, -1)
    M[:12] = 0; M[-12:] = 0; M[:, :12] = 0; M[:, -12:] = 0
    return M
det = cv2.ORB_create(nfeatures=6000, fastThreshold=5)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def motion(n, m):
    g1, g2 = gray(n), gray(m)
    M = mask_for(n, m)
    k1, d1 = det.detectAndCompute(g1, M); k2, d2 = det.detectAndCompute(g2, M)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20: return None
    mt = bf.match(d1, d2)
    if len(mt) < 20: return None
    p1 = np.float32([k1[x.queryIdx].pt for x in mt]); p2 = np.float32([k2[x.trainIdx].pt for x in mt])
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=5000, confidence=0.999)
    if A is None: return None
    s = float(np.sqrt(A[0, 0]**2 + A[1, 0]**2)); rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    return dict(s=s, rot=rot, tx=float(A[0, 2]), ty=float(A[1, 2]), inliers=int(inl.sum()), matches=len(mt))
if __name__ == '__main__':
    res = {}
    for n in range(1, 163):
        m = n+1
        if m in CUTS: continue
        r = motion(n, m)
        res[f'{n}-{m}'] = r
        if r:
            # implied fixed point of the similarity (zoom centre) if |1-s| big enough
            print(f'f{n:03d}->f{m:03d} s={r["s"]:.4f} rot={r["rot"]:+.2f} t=({r["tx"]:+.1f},{r["ty"]:+.1f}) inl={r["inliers"]}/{r["matches"]}')
        else:
            print(f'f{n:03d}->f{m:03d} none')
    json.dump(res, open(f'{CACHE}/bgmotion.json', 'w'), indent=1)
