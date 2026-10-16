"""Round-2 fit: extra candidates, Inter feature variants, emboldened FreeSans Bold (Helvetica proxy)."""
import json, sys
from fonts_fit import *
from fonts_sans_metrics import metrics, REF
A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
refs = prep(SANS_REFS)
# name: (path-or-family, axes, features, strokes)
C = {
 'monasans': ('monasans', {'wght': 700, 'wdth': 100}, None, [0]),
 'hubotsans': ('hubotsans', {'wght': 700, 'wdth': 100}, None, [0]),
 'redditsans': ('redditsans', {'wght': 800}, None, [0]),
 'rethinksans': ('rethinksans', {'wght': 800}, None, [0]),
 'gantari': ('gantari', {'wght': 800}, None, [0]),
 'radiocanadabig': ('radiocanadabig', {'wght': 700}, None, [0]),
 'wixmadefordisplay': ('wixmadefordisplay', {'wght': 800}, None, [0]),
 'redhatdisplay': ('redhatdisplay', {'wght': 800}, None, [0]),
 'roboto': ('roboto', {'wght': 800, 'wdth': 100}, None, [0]),
 'suse': ('suse', {'wght': 800}, None, [0]),
 'bevietnampro-Bold': ('bevietnampro', {}, None, [0]),
 'inter+ss07': ('inter', {'wght': 775, 'opsz': 32}, ['ss07'], [0]),
 'inter-opsz14': ('inter', {'wght': 775, 'opsz': 14}, None, [0]),
 'FreeSansBold+stroke(system,GPL)': ('/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', {}, None, [0.25, 0.5, 0.75, 1.0, 1.25]),
}
only = sys.argv[1:]
res = {}
for name, (fam, axes, feats, strokes) in C.items():
    if only and name not in only: continue
    path = fam if fam.startswith('/') else font_path(fam, 'Bold' if name.endswith('-Bold') else None)
    axes = dict(axes)
    m = metrics(path, axes)
    size, trk = m['size'], (REF['gap_in'] - m['gap_in']) / m['size']
    best = None
    for st in strokes:
        sz, tk, ax = size, trk - 2 * st / size, dict(axes)
        cur = score(path, ax, sz, tk, refs, r=2, features=feats, stroke=st)[0]
        for it in range(2):
            for s in np.arange(0.95, 1.051, 0.01):
                v = score(path, ax, size * s, tk, refs, r=2, features=feats, stroke=st)[0]
                if v > cur + 1e-4: cur, sz = v, size * s
            for dt in np.arange(-0.03, 0.0301, 0.005):
                v = score(path, ax, sz, tk + dt, refs, r=2, features=feats, stroke=st)[0]
                if v > cur + 1e-4: cur, tk = v, tk + dt
            if 'wght' in ax:
                for dw in (-75, -50, -25, 25, 50, 75):
                    a2 = dict(ax); a2['wght'] = ax['wght'] + dw
                    v = score(path, a2, sz, tk, refs, r=2, features=feats, stroke=st)[0]
                    if v > cur + 1e-4: cur, ax = v, a2
        if best is None or cur > best[0]: best = (cur, sz, tk, ax, st)
    cur, sz, tk, ax, st = best
    final, per = score(path, ax, sz, tk, refs, r=3, features=feats, stroke=st)
    res[name] = dict(path=path, axes=ax, features=feats, stroke=st, size=round(float(sz), 2), track_em=round(float(tk), 4),
                     iou=round(float(final), 4), per={w[0]: round(float(p), 3) for w, p in zip(SANS_REFS, per)})
    print(name, res[name], flush=True)
old = json.load(open(A + 'cache_sans_fit.json'))
old.update(res)
json.dump(old, open(A + 'cache_sans_fit.json', 'w'), indent=1)
for k, v in sorted(old.items(), key=lambda kv: -kv[1]['iou']):
    print(f"{v['iou']:.4f}  {k:32s} {v['axes']} size={v['size']} trk={v['track_em']} per={v['per']}")
