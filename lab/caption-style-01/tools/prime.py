"""Solve the 'Prime' / 'at my' compositing with clean background plates from the neighbouring word."""
import numpy as np, cv2, json, sys
from scipy import ndimage as ndi
from PIL import Image
from common import frames, W, H, LAB
a = frames()
CACHE = LAB + '/analysis/cache'
bgm = json.load(open(f'{CACHE}/bgmotion.json'))
def F(n): return a[n-1].astype(np.float32)/255
def sim_matrix(n, m):
    """similarity mapping points of frame n to frame m (chain of consecutive estimates)"""
    A = np.eye(3)
    step = 1 if m > n else -1
    for k in range(n, m, step):
        key = f'{min(k,k+step)}-{max(k,k+step)}'
        r = bgm[key]
        s, th = r['s'], np.radians(r['rot'])
        M = np.array([[s*np.cos(th), -s*np.sin(th), r['tx']], [s*np.sin(th), s*np.cos(th), r['ty']], [0, 0, 1]])
        if step < 0: M = np.linalg.inv(M)
        A = M @ A
    return A
def plate(src, dst):
    """warp frame src into frame dst's geometry"""
    A = sim_matrix(src, dst)
    return cv2.warpAffine(F(src), A[:2].astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def fit_models(I, B):
    """I, B: Nx3 arrays (displayed text pixels, clean background)."""
    out = {}
    # general per-channel affine
    aff = []
    for c in range(3):
        X = np.stack([np.ones(len(B)), B[:, c]], 1)
        coef, *_ = np.linalg.lstsq(X, I[:, c], rcond=None)
        aff.append(coef)
    aff = np.array(aff)
    pred = aff[:, 0] + aff[:, 1]*B
    out['affine_per_channel'] = dict(a=aff[:, 0].round(4).tolist(), b=aff[:, 1].round(4).tolist(), rmse=float(np.sqrt(((I-pred)**2).mean())))
    # H_tint_diff: I = T*(1-B)
    T = ((I*(1-B)).sum(0) / ((1-B)**2).sum(0))
    pred = T*(1-B); out['tint_x_difference'] = dict(T=T.round(4).tolist(), rmse=float(np.sqrt(((I-pred)**2).mean())))
    # H_diff_color: I = |C - B|, grid search C
    best = None
    for r in np.linspace(0, 1, 41):
        pr = np.abs(r - B[:, 0])
        er = ((I[:, 0]-pr)**2).mean()
        if best is None or er < best[0]: best = (er, r)
    Cs = []
    for c in range(3):
        bb = None
        for v in np.linspace(0, 1, 101):
            er = ((I[:, c] - np.abs(v - B[:, c]))**2).mean()
            if bb is None or er < bb[0]: bb = (er, v)
        Cs.append(bb[1])
    Cs = np.array(Cs); pred = np.abs(Cs - B)
    out['difference_with_colour'] = dict(C=Cs.round(3).tolist(), rmse=float(np.sqrt(((I-pred)**2).mean())))
    # normal colour at opacity al: I = (1-al)B + al*C  -> per channel affine with shared al: b = 1-al
    bb = None
    for al in np.linspace(0, 1, 101):
        C = ((I - (1-al)*B).mean(0)) / max(al, 1e-3)
        pred = (1-al)*B + al*np.clip(C, 0, 1)
        er = ((I-pred)**2).mean()
        if bb is None or er < bb[0]: bb = (er, al, C)
    out['normal_colour_opacity'] = dict(alpha=round(float(bb[1]), 3), C=np.clip(bb[2], 0, 1).round(3).tolist(), rmse=float(np.sqrt(bb[0])))
    # multiply with colour
    C = (I*B).sum(0)/(B*B).sum(0); pred = C*B
    out['multiply_colour'] = dict(C=C.round(3).tolist(), rmse=float(np.sqrt(((I-pred)**2).mean())))
    # screen with colour
    bb = None
    for v in np.linspace(0, 1, 51):
        pass
    Cs = []
    for c in range(3):
        b2 = None
        for v in np.linspace(0, 1, 101):
            er = ((I[:, c] - (1-(1-v)*(1-B[:, c])))**2).mean()
            if b2 is None or er < b2[0]: b2 = (er, v)
        Cs.append(b2[1])
    Cs = np.array(Cs); pred = 1-(1-Cs)*(1-B)
    out['screen_colour'] = dict(C=Cs.round(3).tolist(), rmse=float(np.sqrt(((I-pred)**2).mean())))
    # exclusion with colour: I = C + B - 2CB
    Cs = []
    for c in range(3):
        x = 1 - 2*B[:, c]; y = I[:, c] - B[:, c]
        Cs.append(float((x*y).sum()/(x*x).sum()))
    Cs = np.clip(np.array(Cs), 0, 1); pred = Cs + B - 2*Cs*B
    out['exclusion_colour'] = dict(C=Cs.round(3).tolist(), rmse=float(np.sqrt(((I-pred)**2).mean())))
    # difference(white) followed by normal colour layer at opacity beta: I = (1-beta)(1-B) + beta*C
    bb = None
    for be in np.linspace(0, 1, 101):
        C = ((I - (1-be)*(1-B)).mean(0)) / max(be, 1e-3)
        pred = (1-be)*(1-B) + be*np.clip(C, 0, 1)
        er = ((I-pred)**2).mean()
        if bb is None or er < bb[0]: bb = (er, be, C)
    out['difference_then_colour_overlay'] = dict(beta=round(float(bb[1]), 3), C=np.clip(bb[2], 0, 1).round(3).tolist(), rmse=float(np.sqrt(bb[0])))
    # 'color' blend: luminance of (1-B), hue/sat of C (Rec709 lum), fit C by search over hue
    def lum(x): return x[..., 0]*0.2126 + x[..., 1]*0.7152 + x[..., 2]*0.0722
    L = lum(1-B)
    bb = None
    for r in np.linspace(0.5, 1, 26):
        for g in np.linspace(0, 0.6, 31):
            for b in np.linspace(0, 0.7, 36):
                C = np.array([r, g, b])
                d = L[:, None] - lum(C)
                pred = np.clip(C + d, 0, 1)
                er = ((I-pred)**2).mean()
                if bb is None or er < bb[0]: bb = (er, C)
    out['colour_blend_over_inverted'] = dict(C=bb[1].round(3).tolist(), rmse=float(np.sqrt(bb[0])))
    # plain white difference (reference)
    out['white_difference'] = dict(rmse=float(np.sqrt(((I-(1-B))**2).mean())))
    out['white_normal'] = dict(rmse=float(np.sqrt(((I-1)**2).mean())))
    return out

if __name__ == '__main__':
    S = sys.argv[1]
    res = {}
    for name, n, src, excl in [('Prime', 28, 27, (820, 495, 1100, 610)), ('Prime_f29', 29, 27, (820, 495, 1100, 610)), ('at my', 27, 28, None), ('at my_f25', 25, 28, None)]:
        I = F(n); P = plate(src, n)
        D = np.abs(I - P).max(-1)
        # candidate region
        if name.startswith('Prime'):
            R = np.zeros((H, W), bool); R[380:700, 590:1410] = True
            R[excl[1]:excl[3], excl[0]:excl[2]] = False
        else:
            R = np.zeros((H, W), bool); R[500:610, 825:1095] = True
        M = (D > 0.12) & R
        M = ndi.binary_opening(M, iterations=1)
        lab, k = ndi.label(M); sz = ndi.sum(M, lab, range(1, k+1))
        M = np.isin(lab, 1 + np.nonzero(sz >= 60)[0])
        Mi = ndi.binary_erosion(M, iterations=2)
        Ii, Bi = I[Mi], P[Mi]
        r = fit_models(Ii, Bi); r['n_pixels'] = int(Mi.sum())
        r['bg_luma_range_p5_p95'] = [float(np.percentile(Bi.mean(1), 5)), float(np.percentile(Bi.mean(1), 95))]
        res[name] = r
        print(name, json.dumps({k: (v['rmse'] if isinstance(v, dict) and 'rmse' in v else v) for k, v in r.items()}, indent=0))
        # save diagnostic: scatter-ish image of (B luma -> I rgb)
        np.save(f'{CACHE}/prime_mask_{n}.npy', M)
        vis = np.concatenate([I[350:720, 560:1440], P[350:720, 560:1440], np.repeat(M[350:720, 560:1440, None], 3, -1).astype(np.float32)], 0)
        Image.fromarray((np.clip(vis, 0, 1)*255).astype(np.uint8)).save(f'{S}/prime_{name.replace(" ", "_")}.png')
    json.dump(res, open(f'{CACHE}/prime_fit.json', 'w'), indent=1)
    for k, v in res.items():
        print('==', k)
        for m, d in v.items(): print('  ', m, d)
