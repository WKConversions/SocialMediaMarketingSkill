"""Effective text colour per word under I_in = |C - B| (difference with colour C), from edge pairs."""
import numpy as np, json
from blendtest import warped_mask, pairs, reg
from common import frames
from words import WORDS
a = frames()
out = {}
vs = np.linspace(0.5, 1.0, 101)
for wid, txt, f0, f1, rep, box, model in WORDS:
    if wid not in reg: continue
    Ii, Io = [], []
    for n in range(f0, f1+1):
        M, sc = warped_mask(wid, n)
        if sc < 0.6: continue
        I = a[n-1].astype(np.float32)/255
        x, y = pairs(M, I, dmin=2.0, dmax=5.0)
        Ii.append(x); Io.append(y)
    Ii = np.concatenate(Ii); Io = np.concatenate(Io)
    C = []; rm = []
    for c in range(3):
        if model == 'normal':
            er = [np.mean((Ii[:, c] - v)**2) for v in vs]
        else:
            er = [np.mean((Ii[:, c] - np.abs(v - Io[:, c]))**2) for v in vs]
        k = int(np.argmin(er)); C.append(vs[k]); rm.append(np.sqrt(er[k]))
    C = np.array(C)
    out[wid] = dict(text=txt, model=model, C=C.round(3).tolist(), C_rgb255=[int(round(v*255)) for v in C], rmse=round(float(np.mean(rm)), 4),
                    bg_luma_p5_p95=[round(float(np.percentile(Io.mean(1), 5)), 3), round(float(np.percentile(Io.mean(1), 95)), 3)], n=int(len(Ii)))
    print(wid, f'{txt:16s}', model, out[wid]['C_rgb255'], 'rmse', out[wid]['rmse'], 'bg', out[wid]['bg_luma_p5_p95'])
json.dump(out, open('/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache/colourfit.json', 'w'), indent=1)
