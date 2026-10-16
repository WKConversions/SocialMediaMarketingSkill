"""Chroma-flip test: with difference blend at opacity al, chroma(I_in) = (1-2al) * chroma(B).
Uses edge pairs (inside / mirrored outside). Luminance-independent, so robust to grading offsets."""
import numpy as np, json, sys
from blendtest import warped_mask, pairs, reg
from common import frames
from words import WORDS
a = frames()
out = {}
for wid, txt, f0, f1, rep, box, model in WORDS:
    if wid not in reg: continue
    res = {}
    for n in range(f0, f1+1):
        M, sc = warped_mask(wid, n)
        if sc < 0.55: continue
        I = a[n-1].astype(np.float32)/255
        Iin, Iout = pairs(M, I, dmin=2.0, dmax=5.0)
        ci = Iin - Iin.mean(1, keepdims=True); co = Iout - Iout.mean(1, keepdims=True)
        mag = np.linalg.norm(co, axis=1)
        sel = mag > 0.06          # only chromatic background
        if sel.sum() < 50: res[n] = None; continue
        x = co[sel].ravel(); y = ci[sel].ravel()
        k = float((x*y).sum()/(x*x).sum())
        res[n] = dict(chroma_slope=round(k, 3), alpha_from_chroma=round((1-k)/2, 3), n=int(sel.sum()))
    out[wid] = res
    s = ' '.join(f"f{n}:{(r['chroma_slope'] if r else None)}" for n, r in res.items())
    print(f'{wid} {txt:16s} {s}')
json.dump(out, open('/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache/chroma_test.json', 'w'), indent=1)
