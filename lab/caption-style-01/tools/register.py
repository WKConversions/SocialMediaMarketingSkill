"""Per-frame similarity registration (scale + translation) of a word's reference mask
against each frame, using normalized correlation of gradient magnitudes."""
import numpy as np, cv2, json, sys
from scipy import ndimage as ndi
from common import frames, W, H
from words import WORDS, KEEP
CACHE = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
a = frames()

def gradmag_frame(n):
    f = a[n-1].astype(np.float32) / 255
    f = cv2.GaussianBlur(f, (0, 0), 0.8)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2).max(-1)

def ref_full(wid, box, M=None):
    if M is None: M = np.load(f'{CACHE}/ref_{wid}.npy')
    full = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = box
    full[y0:y1, x0:x1] = M
    return full

def register(full_mask, nums, pad=50, smin=0.80, smax=1.25, sstep=0.005):
    A = cv2.GaussianBlur(full_mask, (0, 0), 0.8)
    ys, xs = np.nonzero(full_mask > 0.5)
    cx, cy = xs.mean(), ys.mean()
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    gxA = cv2.Sobel(A, cv2.CV_32F, 1, 0, ksize=3); gyA = cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3)
    GA = np.sqrt(gxA**2 + gyA**2)
    out = []
    for n in nums:
        GT = gradmag_frame(n)
        best = (-2, None)
        res_by_s = {}
        for s in np.arange(smin, smax+1e-9, sstep):
            Mx = np.float32([[s, 0, (1-s)*cx], [0, s, (1-s)*cy]])
            # template: warped GA cropped to warped bbox (+4)
            tx0 = int(np.floor(cx + s*(bx0-cx))) - 4; tx1 = int(np.ceil(cx + s*(bx1-cx))) + 4
            ty0 = int(np.floor(cy + s*(by0-cy))) - 4; ty1 = int(np.ceil(cy + s*(by1-cy))) + 4
            Wp = cv2.warpAffine(GA, Mx, (W, H), flags=cv2.INTER_LINEAR)
            T = Wp[ty0:ty1, tx0:tx1]
            ix0, iy0 = max(0, tx0-pad), max(0, ty0-pad)
            ix1, iy1 = min(W, tx1+pad), min(H, ty1+pad)
            Iimg = GT[iy0:iy1, ix0:ix1]
            R = cv2.matchTemplate(Iimg, T, cv2.TM_CCORR_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(R)
            res_by_s[round(s, 4)] = mx
            if mx > best[0]:
                # subpixel peak
                px, py = loc
                def sub(v_m, v_0, v_p):
                    d = v_m - 2*v_0 + v_p
                    return 0.0 if abs(d) < 1e-9 else 0.5*(v_m - v_p)/d
                ox = sub(R[py, px-1], R[py, px], R[py, px+1]) if 0 < px < R.shape[1]-1 else 0
                oy = sub(R[py-1, px], R[py, px], R[py+1, px]) if 0 < py < R.shape[0]-1 else 0
                dx = (ix0 + px + ox) - tx0; dy = (iy0 + py + oy) - ty0
                best = (mx, (s, dx, dy))
        mx, (s, dx, dy) = best
        # parabolic refine of s
        ks = sorted(res_by_s)
        i = ks.index(round(s, 4))
        if 0 < i < len(ks)-1:
            vm, v0, vp = res_by_s[ks[i-1]], res_by_s[ks[i]], res_by_s[ks[i+1]]
            d = vm - 2*v0 + vp
            if abs(d) > 1e-9: s = s + sstep*0.5*(vm-vp)/d
        ccx, ccy = cx + dx, cy + dy
        out.append(dict(frame=n, score=round(float(mx), 4), scale=round(float(s), 4),
                        cx=round(float(ccx), 2), cy=round(float(ccy), 2),
                        bbox=[round(float(ccx + s*(bx0-cx)), 1), round(float(ccy + s*(by0-cy)), 1),
                              round(float(ccx + s*(bx1-cx)), 1), round(float(ccy + s*(by1-cy)), 1)]))
    return dict(ref_cx=float(cx), ref_cy=float(cy), ref_bbox=[int(bx0), int(by0), int(bx1), int(by1)], frames=out)

if __name__ == '__main__':
    ids = sys.argv[1:]
    allres = {}
    import os
    path = f'{CACHE}/registration.json'
    if os.path.exists(path): allres = json.load(open(path))
    for wid, txt, f0, f1, rep, box, model in WORDS:
        if ids and wid not in ids: continue
        if not os.path.exists(f'{CACHE}/ref_{wid}.npy'): continue
        full = ref_full(wid, box)
        r = register(full, list(range(max(1, f0-1), min(163, f1+1)+1)))
        r['text'] = txt; r['rep'] = rep
        allres[wid] = r
        print(wid, txt, 'ref bbox', r['ref_bbox'])
        for fr in r['frames']:
            print(f"  f{fr['frame']:03d} score={fr['score']:.3f} s={fr['scale']:.4f} c=({fr['cx']:.1f},{fr['cy']:.1f}) bbox={fr['bbox']}")
    json.dump(allres, open(path, 'w'), indent=1)
