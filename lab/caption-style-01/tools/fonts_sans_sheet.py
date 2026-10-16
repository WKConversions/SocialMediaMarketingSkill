"""Build sans comparison sheets: reference crops on top, candidates below (white on black),
each candidate scaled to the reference x-height and tracked to the reference stem gap."""
import sys, json
from fonts_lib import *
from fonts_sans_metrics import metrics, REF

CX, BASE = 958.0, 579.5          # word centre x, baseline y in the 1916x1078 frame
WORDS = {'charging': 78, 'winning': 142, 'yet': 20, 'still': 67, 'and': 146, 'at a': 46, 'know': 124}


def tile_ref(word, x0, x1, y0, y1, zoom):
    im = ref_crop(WORDS[word], (x0, y0, x1, y1))
    return im.resize((int(im.width * zoom), int(im.height * zoom)), Image.LANCZOS)


def tile_cand(path, axes, size, track_em, word, x0, x1, y0, y1, zoom):
    f = load(path, size * zoom, axes)
    tr = track_em * size * zoom
    # layout per glyph with tracking, measure total ink, centre on CX
    W, H = int((x1 - x0) * zoom), int((y1 - y0) * zoom)
    im = Image.new('L', (W * 3, H), 0)
    d = ImageDraw.Draw(im)
    pen = W * 1.0
    xs = []
    for i, ch in enumerate(word):
        xs.append(pen)
        if i + 1 < len(word):
            pair = f.getlength(word[i:i + 2]); nxt = f.getlength(word[i + 1])
            pen += (pair - nxt) + tr
    by = (BASE - y0) * zoom
    for x, ch in zip(xs, word):
        d.text((x, by), ch, font=f, fill=255, anchor='ls')
    a = np.asarray(im); cols = np.nonzero(a.max(axis=0) > 60)[0]
    cx = (cols.min() + cols.max() + 1) / 2
    shift = (CX - x0) * zoom - cx
    out = Image.new('L', (W, H), 0)
    out.paste(im, (int(round(shift)), 0))
    return out


def build(cands, words, out, zoom=1.0, boxes=None):
    boxes = boxes or {}
    cols = []
    for w in words:
        x0, x1, y0, y1 = boxes.get(w, (700, 1216, 470, 620))
        tiles = [label(tile_ref(w, x0, x1, y0, y1, zoom), f'REFERENCE  "{w}"  f{WORDS[w]:03d}')]
        for name, path, axes, size, trk in cands:
            t = tile_cand(path, axes, size, trk, w, x0, x1, y0, y1, zoom)
            ax = ','.join(f'{k}={v}' for k, v in axes.items())
            tiles.append(label(t, f'{name}  {ax}  {size:.0f}px  trk {trk:+.3f}em'))
        cols.append(stack(tiles, gap=4))
    sheet = Image.new('RGB', (sum(c.width for c in cols) + 8 * (len(cols) - 1), max(c.height for c in cols)), (60, 60, 60))
    x = 0
    for c in cols:
        sheet.paste(c, (x, 0)); x += c.width + 8
    sheet.save(out)
    print('saved', out, sheet.size)


def auto(fam, axes, pick=None):
    p = font_path(fam, pick)
    m = metrics(p, axes)
    trk = (REF['gap_in'] - m['gap_in']) / m['size']
    return (fam, p, axes, m['size'], round(trk, 3))
