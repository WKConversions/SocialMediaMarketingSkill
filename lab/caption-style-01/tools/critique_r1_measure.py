"""Measure text ink geometry + difference-blend accuracy on near-unzoomed frames."""
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
S = Path(sys.argv[1])
LAB = Path('/home/user/SocialMediaMarketingSkill/lab/caption-style-01')
rep = json.load(open(LAB / 'out/user_v1.render.json'))
def load(p): return np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255
for n in [int(x) for x in sys.argv[2].split(',')]:
    fr = rep['frames'][n]
    o = load(S / f'ours/o{n:03d}.png'); s = load(S / f'src/s{fr["source_frame"]:03d}.png')
    words = '+'.join(w['text'] for w in fr['words'])
    inv = 1 - s
    d_src = np.abs(o - s).mean(2); d_inv = np.abs(o - inv).mean(2)
    # text pixels: where output is much closer to inverted source than to source
    m = (d_inv < 0.06) & (d_src > 0.25)
    ys, xs = np.nonzero(m)
    outside = ~m
    # dilate approx by excluding a box around text for 'untouched' stats
    box = fr['words'][0]['box'] if fr['words'] else None
    ok = np.ones(m.shape, bool)
    if box: ok[max(0, box[1]-40):box[3]+40, max(0, box[0]-40):box[2]+40] = False
    untouched = np.abs(o - s)[ok]
    print(f'f{n:03d} z{fr["zoom"]:.4f} {words:10s} textpx={m.sum():6d} ink x[{xs.min() if len(xs) else -1},{xs.max() if len(xs) else -1}] y[{ys.min() if len(ys) else -1},{ys.max() if len(ys) else -1}]'
          f'  untouched |o-s| mean={untouched.mean()*255:.2f} p99={np.percentile(untouched,99)*255:.1f}'
          f'  per-channel mean(o-s) outside = {[round(float(v)*255,2) for v in (o-s)[ok].mean(0)]}')
    if box:
        b = np.zeros(m.shape, bool); b[box[1]:box[3], box[0]:box[2]] = True
        # inside box: pixels that are text = closer to inv than src
        tb = b & (d_inv < d_src)
        e = np.abs(o - inv)[tb]
        print(f'      in-box text px {tb.sum()} |o-(1-s)| median={np.median(e)*255:.2f} p90={np.percentile(e,90)*255:.1f}')
