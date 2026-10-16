"""Round-1 numeric screen of sans candidates against reference proportions."""
import numpy as np
from fonts_lib import *

REF = dict(xh=58.5, stem=17.5, nw=56.0, gap_in=3.0, gap_ll=3.0, asc=80.0, desc=22.5)
S = 400  # render size for metrics


def stem_width(font, ch='l'):
    im = Image.new('L', (S * 2, S * 2), 0)
    d = ImageDraw.Draw(im)
    d.text((S // 2, S), ch, font=font, fill=255, anchor='ls')
    a = np.asarray(im) > 127
    xh = font.getbbox('x', anchor='ls')
    y = S - int(0.5 * (-xh[1]))
    row = a[y]
    xs = np.nonzero(row)[0]
    # first run width
    runs = []
    start = None
    for i, v in enumerate(row):
        if v and start is None: start = i
        if not v and start is not None: runs.append(i - start); start = None
    return runs[0] if runs else 0


def metrics(path, axes):
    f = load(path, S, axes)
    xb = f.getbbox('x', anchor='ls'); xh = -xb[1]
    lb = f.getbbox('l', anchor='ls'); asc = -lb[1]
    gb = f.getbbox('g', anchor='ls'); desc = gb[3]
    stem = stem_width(f, 'l')
    n = glyph_metrics(f, 'n'); i = glyph_metrics(f, 'i'); l = glyph_metrics(f, 'l')
    k = xh / REF['xh']  # scale factor render px -> ref px is 1/k
    return dict(stem=stem / k, nw=n['w'] / k, asc=asc / k, desc=desc / k,
                gap_in=(i['rsb'] + n['lsb']) / k, gap_ll=(l['rsb'] + l['lsb']) / k,
                size=S / k)

CANDS_SANS = {
    'inter': ['Inter['], 'intertight': None, 'manrope': None, 'plusjakartasans': None,
    'instrumentsans': None, 'figtree': None, 'geist': None, 'onest': None, 'archivo': None,
    'schibstedgrotesk': None, 'hankengrotesk': None, 'albertsans': None, 'publicsans': None,
    'hostgrotesk': None, 'golostext': None, 'arimo': None, 'funnelsans': None,
    'familjengrotesk': None, 'worksans': None,
}

if __name__ == '__main__':
    print('REF', REF)
    for fam in CANDS_SANS:
        p = font_path(fam)
        ax = axes_info(p)
        names = [a[0] for a in ax]
        wmin = min([a[1] for a in ax if a[0] in ('Weight',)] or [400])
        wmax = max([a[3] for a in ax if a[0] in ('Weight',)] or [400])
        best = None
        for w in range(int(wmin), int(wmax) + 1, 25):
            axes = {'wght': w}
            if fam == 'inter': axes['opsz'] = 32
            m = metrics(p, axes)
            if best is None or abs(m['stem'] - REF['stem']) < abs(best[1]['stem'] - REF['stem']):
                best = (w, m)
        w, m = best
        trk = (REF['gap_in'] - m['gap_in']) / m['size']
        print(f"{fam:18s} wght={w:4d} size={m['size']:.1f} stem={m['stem']:.1f} nw={m['nw']:.1f} asc={m['asc']:.1f} desc={m['desc']:.1f} gap_in={m['gap_in']:.1f} gap_ll={m['gap_ll']:.1f} track_em={trk:+.3f}  axes={names}")
