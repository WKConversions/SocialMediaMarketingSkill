"""Edge-pair blend-mode test. For pixels just inside the glyph and their mirror just outside,
fit I_in = a + b*I_out per channel.
  difference (white, opacity al):  a = al, b = 1-2al   (al=1 -> a=1, b=-1)
  normal     (white, opacity al):  a = al, b = 1-al    (al=1 -> a=1, b=0)
Also fit alpha under each model and compare RMSE."""
import numpy as np, cv2, json, sys
from scipy import ndimage as ndi
from common import frames, W, H
from words import WORDS
CACHE = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
a = frames()
reg = json.load(open(f'{CACHE}/registration.json'))

def warped_mask(wid, n, box=None):
    r = reg[wid]
    fr = [f for f in r['frames'] if f['frame'] == n][0]
    M = np.load(f'{CACHE}/ref_{wid}.npy').astype(np.float32)
    if box is None: box = [w for w in WORDS if w[0] == wid][0][5]
    full = np.zeros((H, W), np.float32); x0, y0, x1, y1 = box; full[y0:y1, x0:x1] = M
    s = fr['scale']; cx, cy = r['ref_cx'], r['ref_cy']; tx, ty = fr['cx'], fr['cy']
    A = np.float32([[s, 0, tx - s*cx], [0, s, ty - s*cy]])
    return cv2.warpAffine(full, A, (W, H), flags=cv2.INTER_LINEAR) > 0.5, fr['score']

def bilinear(img, xs, ys):
    return np.stack([cv2.remap(img[..., c], xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR) for c in range(3)], -1)

def pairs(M, I, dmin=2.5, dmax=6.0, roi=None):
    din, idx = ndi.distance_transform_edt(M, return_indices=True)
    dout = ndi.distance_transform_edt(~M)
    sel = (din >= dmin) & (din <= dmax)
    if roi is not None: sel &= roi
    py, px = np.nonzero(sel)
    by, bx = idx[0][py, px], idx[1][py, px]          # nearest background pixel
    vy, vx = by - py, bx - px; L = np.sqrt(vx**2 + vy**2) + 1e-6
    d = din[py, px]
    qy = py + vy/L*(2*d); qx = px + vx/L*(2*d)        # mirror point at same distance outside
    ok = (qx >= 0) & (qx < W-1) & (qy >= 0) & (qy < H-1)
    py, px, qy, qx = py[ok], px[ok], qy[ok], qx[ok]
    ok2 = dout[np.round(qy).astype(int), np.round(qx).astype(int)] >= dmin-1
    py, px, qy, qx = py[ok2], px[ok2], qy[ok2], qx[ok2]
    Iin = I[py, px]
    Iout = bilinear(I, qx[None, :], qy[None, :])[0]
    return Iin, Iout

def fit(Iin, Iout):
    res = {}
    # per-channel linear fit
    ab = []
    for c in range(3):
        X = np.stack([np.ones_like(Iout[:, c]), Iout[:, c]], 1)
        coef, *_ = np.linalg.lstsq(X, Iin[:, c], rcond=None)
        ab.append([round(float(coef[0]), 3), round(float(coef[1]), 3)])
    res['lin_a_b_rgb'] = ab
    y = (Iin - Iout).ravel()
    xd = (1 - 2*Iout).ravel(); xn = (1 - Iout).ravel()
    ald = float((xd*y).sum() / (xd*xd).sum()); aln = float((xn*y).sum() / (xn*xn).sum())
    res['alpha_diff'] = round(ald, 3); res['alpha_normal'] = round(aln, 3)
    res['rmse_diff'] = round(float(np.sqrt(((y - ald*xd)**2).mean())), 4)
    res['rmse_normal'] = round(float(np.sqrt(((y - aln*xn)**2).mean())), 4)
    res['rmse_diff_alpha1'] = round(float(np.sqrt(((Iin - (1 - Iout))**2).mean())), 4)
    res['rmse_normal_alpha1'] = round(float(np.sqrt(((Iin - 1)**2).mean())), 4)
    res['n_pairs'] = int(len(Iin))
    # correlation between inside and outside luminance (difference -> strongly negative)
    li = Iin.mean(1); lo = Iout.mean(1)
    res['corr_in_out_luma'] = round(float(np.corrcoef(li, lo)[0, 1]), 3) if lo.std() > 1e-3 else None
    res['out_luma_range'] = [round(float(np.percentile(lo, 5)), 3), round(float(np.percentile(lo, 95)), 3)]
    return res

if __name__ == '__main__':
    ids = sys.argv[1:]
    out = {}
    for wid, txt, f0, f1, rep, box, model in WORDS:
        if ids and wid not in ids: continue
        if wid not in reg: continue
        allin, allout = [], []
        per = {}
        for n in range(f0, f1+1):
            M, sc = warped_mask(wid, n)
            if sc < 0.6: continue
            I = a[n-1].astype(np.float32)/255
            Iin, Iout = pairs(M, I)
            per[n] = fit(Iin, Iout)
            allin.append(Iin); allout.append(Iout)
        tot = fit(np.concatenate(allin), np.concatenate(allout))
        out[wid] = dict(text=txt, overall=tot, per_frame=per)
        print(f"{wid} {txt:10s} aD={tot['alpha_diff']:.3f} rmseD={tot['rmse_diff']:.3f} | aN={tot['alpha_normal']:.3f} rmseN={tot['rmse_normal']:.3f} | corr={tot['corr_in_out_luma']} lin={tot['lin_a_b_rgb']} n={tot['n_pairs']}")
        for n, p in per.items():
            print(f"     f{n:03d} aD={p['alpha_diff']:.3f} rmseD={p['rmse_diff']:.3f} aN={p['alpha_normal']:.3f} rmseN={p['rmse_normal']:.3f} corr={p['corr_in_out_luma']}")
    json.dump(out, open(f'{CACHE}/blendtest.json', 'w'), indent=1)
