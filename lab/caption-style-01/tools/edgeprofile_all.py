import numpy as np, json
from edgeprofile import profile, fit
from blendtest import warped_mask, reg
from common import frames
from words import WORDS
a = frames()
C = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
out = {}
for wid, txt, f0, f1, rep, box, model in WORDS:
    if wid not in reg or wid == 'w15': continue
    script = wid in ('w10', 'w20', 'w28', 'w06')
    out[wid] = {}
    rows = []
    for n in range(f0, f1+1):
        M, sc = warped_mask(wid, n)
        prof, cnt = profile(a[n-1].astype(np.float32)/255, M.astype(np.float32), minwidth=8 if script else 12)
        if cnt < 50: rows.append((n, None)); continue
        (al, d0, sg), rm = fit(prof)
        out[wid][n] = dict(alpha=round(float(al), 3), sigma=round(float(sg), 2), n=cnt)
        rows.append((n, (al, sg, cnt)))
    print(wid, f'{txt:10s}', ' '.join(f'f{n}:' + (f'{r[0]:.2f}/{r[1]:.1f}' if r else 'NA') for n, r in rows))
json.dump(out, open(f'{C}/edgeprofile_all.json', 'w'), indent=1)
