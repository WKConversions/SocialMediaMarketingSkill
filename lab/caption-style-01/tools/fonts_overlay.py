"""Overlay sheets: reference text mask in red, candidate render in cyan (white = agreement)."""
import json, sys
from fonts_fit import *


def overlay(refm, cand, zoom=2):
    r = (refm * 255).astype(np.uint8)
    c = (cand * 255).astype(np.uint8)
    rgb = np.stack([r, c, c], axis=2)
    im = Image.fromarray(rgb)
    return im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)


def sheet(cands, refs, out, zoom=2, base_y=579.5):
    cols = []
    for word, n, box, m, cx in refs:
        tiles = [label(overlay(m, m * 0, zoom), f'REF mask "{word}" f{n:03d}')]
        for name, path, axes, size, trk, feats in cands:
            rm = render_mask(path, axes, size, trk, word, box, base_y, cx, feats)
            s, dx, dy = iou(m, rm, 3)
            rm = np.roll(np.roll(rm, dy, 0), dx, 1)
            tiles.append(label(overlay(m, rm, zoom), f'{name} IoU {s:.3f}'))
        cols.append(stack(tiles, gap=4))
    W = sum(c.width for c in cols) + 8 * (len(cols) - 1)
    sh = Image.new('RGB', (W, max(c.height for c in cols)), (60, 60, 60))
    x = 0
    for c in cols:
        sh.paste(c, (x, 0)); x += c.width + 8
    sh.save(out); print('saved', out, sh.size)
