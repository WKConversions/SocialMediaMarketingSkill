"""Proportions of script candidates vs reference: stem/x-height, hairline/x, asc/x, slant."""
import sys
from fonts_lib import *
from fonts_script_fit import ref_x, fam_path, SCRIPT_FAMS


def props(path, axes=None, stroke=0):
    S = 300
    xh, asc = ref_x(path)
    f = load(path, S / xh, axes)  # x-height = 300 px
    im = Image.new('L', (3000, 1400), 0); d = ImageDraw.Draw(im)
    d.text((200, 1000), 'nunu', font=f, fill=255, anchor='ls', stroke_width=stroke, stroke_fill=255)
    a = np.asarray(im) > 127
    row = a[1000 - 150]
    runs = []; st = None
    for i, v in enumerate(row):
        if v and st is None: st = i
        if not v and st is not None: runs.append(i - st); st = None
    runs = sorted(runs)
    stem = np.median(runs[-4:]) if len(runs) >= 4 else max(runs)
    # slant: centroid of the first thick run at 25% and 75% x-height
    def first_thick(y):
        row = a[y]; st = None
        for i, v in enumerate(row):
            if v and st is None: st = i
            if not v and st is not None:
                if i - st > stem * 0.6: return (st + i) / 2
                st = None
    x1, x2 = first_thick(1000 - 75), first_thick(1000 - 225)
    slant = np.degrees(np.arctan2(x2 - x1, 150)) if x1 and x2 else float('nan')
    # cap heights for P, W, R, T (ink top above baseline / x-height)
    caps = {c: ink(f, c)[1] / S for c in 'PWRT'}
    return stem / S, asc / xh, slant, caps


if __name__ == '__main__':
    print('REF: stem/x=0.217 (horizontal run, 18/83), asc/x=1.85, slant~30deg, cap/x: P~2.5 W~2.4 R~2.6 T~2.6 (incl. loops)')
    specs = sys.argv[1:] or SCRIPT_FAMS
    for spec in specs:
        p = fam_path(spec)
        s, ax, sl, caps = props(p, {'opsz': 72} if 'Ballet' in p else None)
        print(f'{spec:22s} stem/x={s:.3f} asc/x={ax:.2f} slant={sl:5.1f}  caps ' + ' '.join(f'{k}={v:.2f}' for k, v in caps.items()))
