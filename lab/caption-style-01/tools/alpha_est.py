"""Per-frame opacity / blend estimate using a local background inpainted from just outside the glyphs.
diff model:   I = B + al*(1-2B)      normal model: I = B + al*(1-B)"""
import numpy as np, cv2, json, sys
from scipy import ndimage as ndi
from common import frames, W, H
from words import WORDS
from blendtest import warped_mask, reg
from textmask import nconv
CACHE = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
a = frames()

def estimate(M, I, sigma=7, erode=2, ring=(2, 9)):
    x0, y0, x1, y1 = bbox = cv2.boundingRect(M.astype(np.uint8))[0:4]
    x, y, w, h = bbox
    X0, Y0, X1, Y1 = max(0, x-40), max(0, y-40), min(W, x+w+40), min(H, y+h+40)
    m = M[Y0:Y1, X0:X1]; Ic = I[Y0:Y1, X0:X1]
    dout = ndi.distance_transform_edt(~m)
    ext = (dout >= ring[0]) & (dout <= ring[1])
    B = nconv(Ic, ext.astype(np.float32), sigma)
    inn = ndi.binary_erosion(m, iterations=erode)
    Bi, Ii = B[inn], Ic[inn]
    res = {}
    for name, den in (('diff', 1 - 2*Bi), ('normal', 1 - Bi)):
        num = Ii - Bi
        wgt = np.abs(den)
        ok = wgt > 0.25
        if ok.sum() < 30:
            res[name] = None; continue
        al = (num[ok] / den[ok])
        # weighted median
        o = np.argsort(al); cw = np.cumsum(wgt[ok][o]); med = al[o][np.searchsorted(cw, cw[-1]/2)]
        pred = Bi + med*den
        res[name] = dict(alpha=round(float(med), 3), rmse=round(float(np.sqrt(((Ii - pred)**2).mean())), 4),
                         rmse_full=round(float(np.sqrt(((Ii - (Bi + den))**2).mean())), 4),
                         n=int(ok.sum()))
    res['bg_luma_p5_p95'] = [round(float(np.percentile(Bi.mean(-1), 5)), 3), round(float(np.percentile(Bi.mean(-1), 95)), 3)]
    return res

if __name__ == '__main__':
    ids = sys.argv[1:]
    out = {}
    for wid, txt, f0, f1, rep, box, model in WORDS:
        if ids and wid not in ids: continue
        if wid not in reg: continue
        out[wid] = {}
        print(wid, txt)
        for fr in reg[wid]['frames']:
            n = fr['frame']
            M, sc = warped_mask(wid, n)
            I = a[n-1].astype(np.float32)/255
            r = estimate(M, I); r['reg_score'] = fr['score']
            out[wid][n] = r
            d, nm = r['diff'], r['normal']
            ds = f"aD={d['alpha']:.3f} rmseD={d['rmse']:.3f} rmseD1={d['rmse_full']:.3f}" if d else 'aD=None'
            ns = f"aN={nm['alpha']:.3f} rmseN={nm['rmse']:.3f} rmseN1={nm['rmse_full']:.3f}" if nm else 'aN=None'
            print(f"   f{n:03d} reg={fr['score']:.2f} {ds} | {ns} | bgL={r['bg_luma_p5_p95']}")
    json.dump(out, open(f'{CACHE}/alpha_est.json', 'w'), indent=1)
