"""Round-2 script fit on 'hundred' f055 with extra synthetic slant (shear) and emboldening."""
import json, sys, itertools
from fonts_fit import *
from fonts_script_fit import BOX, BASE, REF_XH, ref_x, fam_path

A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'


def render_script(path, axes, size, trk, word, box, base_y, cx, shear_deg=0.0, stroke=0.0, features=None):
    rm = render_mask(path, axes, size, trk, word, box, base_y, cx, features, stroke)
    if shear_deg:
        k = np.tan(np.radians(shear_deg))
        by = base_y - box[1]
        im = Image.fromarray((rm * 255).astype(np.uint8))
        im = im.transform(im.size, Image.AFFINE, (1, k, -k * by, 0, 1, 0), resample=Image.BICUBIC)
        rm = np.asarray(im).astype(np.float32) / 255
        # re-centre horizontally on cx
        cols = np.nonzero(rm.max(axis=0) > 0.25)[0]
        c = (cols.min() + cols.max() + 1) / 2
        rm = np.roll(rm, int(round((cx - box[0]) - c)), axis=1)
    return rm


def refs_hundred(frame=55):
    m = ref_mask(frame, BOX, 0.3, 0.6)
    cols = np.nonzero((m > 0.5).sum(0) >= 2)[0]
    cx = BOX[0] + (cols.min() + cols.max() + 1) / 2
    return m, cx


def fitone(spec, shears=(0, 3, 6), strokes=(0,), feats=None):
    p = fam_path(spec)
    axes = {'opsz': 72} if 'Ballet' in p else None
    m, cx = refs_hundred()
    xh, _ = ref_x(p)
    size0 = REF_XH / xh
    best = (0,)
    for sh, st in itertools.product(shears, strokes):
        for s in np.arange(0.88, 1.121, 0.02):
            for t in (-0.02, -0.01, 0.0, 0.01, 0.02):
                rm = render_script(p, axes, size0 * s, t, 'hundred', BOX, BASE, cx, sh, st, feats)
                v = iou(m, rm, 4)[0]
                if v > best[0]: best = (v, size0 * s, t, sh, st)
    v, size, t, sh, st = best
    for ds in np.arange(-0.015, 0.0151, 0.005):
        for dt in np.arange(-0.01, 0.0101, 0.0025):
            for dsh in (-1.5, 0, 1.5):
                rm = render_script(p, axes, size * (1 + ds), t + dt, 'hundred', BOX, BASE, cx, sh + dsh, st, feats)
                vv = iou(m, rm, 4)[0]
                if vv > best[0]: best = (vv, size * (1 + ds), t + dt, sh + dsh, st)
    return dict(path=p, axes=axes, iou=round(float(best[0]), 4), size=round(float(best[1]), 1), track_em=round(float(best[2]), 4),
                shear_deg=float(best[3]), stroke=float(best[4]))


if __name__ == '__main__':
    out = {}
    try: out = json.load(open(A + 'cache_script_fit2.json'))
    except Exception: pass
    for spec in sys.argv[1:]:
        r = fitone(spec, strokes=(0, 1, 2))
        out[spec] = r
        print(spec, r, flush=True)
    json.dump(out, open(A + 'cache_script_fit2.json', 'w'), indent=1)
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]['iou']):
        print(f"{v['iou']:.4f} {k:20s} size={v['size']} trk={v['track_em']} shear={v['shear_deg']} stroke={v['stroke']}")
