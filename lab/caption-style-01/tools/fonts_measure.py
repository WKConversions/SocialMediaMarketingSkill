"""Measure caption glyph metrics in reference frames (text bright on dark bg).

usage: python3 fonts_measure.py <frame> <x0> <y0> <x1> <y1> [thresh]
prints the bbox of bright pixels and per-row / per-column ink counts so baseline,
x-height, ascender and descender lines, and letter gaps can be read off.
"""
import sys
import numpy as np
from PIL import Image

FR = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/ref/frames'


def mask(n, box, th=0.55):
    im = np.asarray(Image.open(f'{FR}/f{n:03d}.png').convert('RGB')).astype(np.float32) / 255
    x0, y0, x1, y1 = box
    c = im[y0:y1, x0:x1]
    lum = c.min(axis=2)  # white text: all channels high
    return lum > th


def bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


if __name__ == '__main__':
    n = int(sys.argv[1]); box = tuple(int(v) for v in sys.argv[2:6])
    th = float(sys.argv[6]) if len(sys.argv) > 6 else 0.55
    m = mask(n, box, th)
    x0, y0, x1, y1 = bbox(m)
    print('bbox abs', x0 + box[0], y0 + box[1], x1 + box[0], y1 + box[1], 'w', x1 - x0 + 1, 'h', y1 - y0 + 1)
    rows = m.sum(axis=1)
    for y in range(y0, y1 + 1):
        print(y + box[1], int(rows[y]))
