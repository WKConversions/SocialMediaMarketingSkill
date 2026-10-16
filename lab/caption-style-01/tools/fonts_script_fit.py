"""Numeric screen of script candidates on 'hundred' (f055, light text on dark background):
size from x-height, then search size/tracking for best soft IoU."""
import json, sys, glob, os
from fonts_fit import *

A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
REF_XH, REF_ASC, BASE = 83.0, 153.5, 631.5
BOX = (600, 455, 1380, 650)
SCRIPT_FAMS = ['pinyonscript', 'greatvibes', 'italianno', 'allura', 'alexbrush', 'parisienne', 'monsieurladoulaise',
               'rougescript', 'imperialscript', 'ballet', 'tangerine:Bold', 'tangerine:Regular', 'petitformalscript',
               'mrssaintdelafield', 'herrvonmuellerhoff', 'corinthia:Bold', 'corinthia:Regular', 'ephesis', 'luxuriousscript',
               'birthstone', 'missfajardose', 'arizonia', 'mrdehaviland', 'bonheurroyale', 'fleurdeleah', 'lovelight',
               'montecarlo', 'waterfall', 'lavishlyyours', 'ingriddarling', 'inspiration', 'kolkerbrush', 'puppiesplay',
               'meaculpa', 'islandmoments', 'thenautigal:Bold', 'windsong:Medium', 'euphoriascript', 'engagement',
               'princesssofia', 'loversquarrel']


def fam_path(spec):
    fam, _, pick = spec.partition(':')
    return font_path(fam, pick or None)


def ref_x(path):
    f = load(path, 400, {'opsz': 72} if 'Ballet' in path else None)
    ux = ink(f, 'u')[1]; nx = ink(f, 'n')[1]
    hx = ink(f, 'h')[1]
    return min(ux, nx) / 400, hx / 400


def fit(path, refs, axes=None):
    xh, asc = ref_x(path)
    size0 = REF_XH / xh
    best = (0, size0, 0.0)
    for s in np.arange(0.85, 1.151, 0.025):
        for t in (-0.02, 0.0, 0.02):
            v = score(path, axes, size0 * s, t, refs, base_y=BASE, r=4)[0]
            if v > best[0]: best = (v, size0 * s, t)
    v0, size, trk = best
    for s in np.arange(-0.02, 0.0201, 0.005):
        for t in np.arange(-0.03, 0.0301, 0.005):
            v = score(path, axes, size * (1 + s), trk + t, refs, base_y=BASE, r=4)[0]
            if v > best[0]: best = (v, size * (1 + s), trk + t)
    return best, xh, asc


if __name__ == '__main__':
    refs = []
    m = ref_mask(55, BOX, 0.3, 0.6)
    cols = np.nonzero((m > 0.5).sum(0) >= 2)[0]
    cx = BOX[0] + (cols.min() + cols.max() + 1) / 2
    refs = [('hundred', 55, BOX, m, cx)]
    out = {}
    specs = sys.argv[1:] or SCRIPT_FAMS
    for spec in specs:
        p = fam_path(spec)
        axes = {'opsz': 72} if 'Ballet' in p else None
        (v, size, trk), xh, asc = fit(p, refs, axes)
        out[spec] = dict(path=p, axes=axes, size=round(float(size), 1), track_em=round(float(trk), 3), iou=round(float(v), 4),
                         asc_ratio=round(asc / xh, 2), xh_em=round(xh, 3))
        print(f'{v:.4f} {spec:22s} size={size:.1f} trk={trk:+.3f} asc/x={asc/xh:.2f} (ref {REF_ASC/REF_XH:.2f})', flush=True)
    old = {}
    try: old = json.load(open(A + 'cache_script_fit.json'))
    except Exception: pass
    old.update(out)
    json.dump(old, open(A + 'cache_script_fit.json', 'w'), indent=1)
    print('---')
    for k, v in sorted(old.items(), key=lambda kv: -kv[1]['iou']):
        print(f"{v['iou']:.4f}  {k:22s} size={v['size']} trk={v['track_em']} asc/x={v['asc_ratio']}")
