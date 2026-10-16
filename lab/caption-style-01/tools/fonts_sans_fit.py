import json, sys
from fonts_fit import *
from fonts_sans_metrics import metrics, REF

A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
refs = prep(SANS_REFS)
C = {'inter': (725, {'opsz': 32}), 'intertight': (750, {}), 'manrope': (800, {}), 'plusjakartasans': (800, {}),
     'instrumentsans': (700, {'wdth': 100}), 'figtree': (775, {}), 'geist': (750, {}), 'onest': (800, {}),
     'archivo': (775, {'wdth': 100}), 'schibstedgrotesk': (750, {}), 'hankengrotesk': (875, {}),
     'albertsans': (825, {}), 'publicsans': (750, {}), 'hostgrotesk': (800, {}), 'golostext': (700, {}),
     'arimo': (700, {}), 'funnelsans': (800, {}), 'familjengrotesk': (700, {}), 'worksans': (625, {}),
     'FreeSansBold(system,GPL)': (None, {})}
only = sys.argv[1:]
res = {}
for fam, (w0, extra) in C.items():
    if only and fam not in only: continue
    if fam.startswith('FreeSans'):
        path = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
    else:
        path = font_path(fam)
    axes = dict(extra)
    if w0: axes['wght'] = w0
    m = metrics(path, axes)
    size, trk = m['size'], (REF['gap_in'] - m['gap_in']) / m['size']
    cur = score(path, axes, size, trk, refs, r=2)[0]
    for it in range(2):
        for s in np.arange(0.97, 1.031, 0.01):
            v = score(path, axes, size * s, trk, refs, r=2)[0]
            if v > cur + 1e-4: cur, size_new = v, size * s
            else: size_new = None
            if size_new: size = size_new
        for dt in np.arange(-0.02, 0.0201, 0.005):
            v = score(path, axes, size, trk + dt, refs, r=2)[0]
            if v > cur + 1e-4: cur, trk = v, trk + dt
        if w0:
            for dw in (-50, -25, 25, 50):
                ax2 = dict(axes); ax2['wght'] = axes['wght'] + dw
                v = score(path, ax2, size, trk, refs, r=2)[0]
                if v > cur + 1e-4: cur, axes = v, ax2
    final, per = score(path, axes, size, trk, refs, r=3)
    res[fam] = dict(path=path, axes=axes, size=round(float(size), 2), track_em=round(float(trk), 4), iou=round(float(final), 4),
                    per={w[0]: round(float(p), 3) for w, p in zip(SANS_REFS, per)})
    print(fam, res[fam], flush=True)
old = {}
try: old = json.load(open(A + 'cache_sans_fit.json'))
except Exception: pass
old.update(res)
json.dump(old, open(A + 'cache_sans_fit.json', 'w'), indent=1)
for k, v in sorted(old.items(), key=lambda kv: -kv[1]['iou']):
    print(f"{v['iou']:.4f}  {k:26s} {v['axes']} size={v['size']} trk={v['track_em']}")
