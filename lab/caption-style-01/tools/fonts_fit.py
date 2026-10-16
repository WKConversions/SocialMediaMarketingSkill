"""Objective fit: render a candidate word over the reference crop geometry and score soft IoU
against the reference text mask (bright text on dark background frames only).
Searches weight / size / tracking per candidate (coordinate descent)."""
import json, sys
import numpy as np
from fonts_lib import *

SS = 4  # supersampling


def ref_mask(n, box, lo=0.35, hi=0.65):
    a = np.asarray(ref_crop(n, box)).astype(np.float32) / 255
    v = a.min(axis=2)
    return np.clip((v - lo) / (hi - lo), 0, 1)


def render_mask(path, axes, size, track_em, word, box, base_y, cx, features=None, stroke=0.0):
    x0, y0, x1, y1 = box
    W, H = (x1 - x0), (y1 - y0)
    f = load(path, size * SS, axes)
    tr = track_em * size * SS
    im = Image.new('L', (W * SS * 3, H * SS), 0)
    d = ImageDraw.Draw(im)
    pen = W * SS * 1.0
    xs = []
    for i, ch in enumerate(word):
        xs.append(pen)
        if i + 1 < len(word):
            pair = f.getlength(word[i:i + 2], features=features); nxt = f.getlength(word[i + 1], features=features)
            pen += (pair - nxt) + tr
    by = (base_y - y0) * SS
    for x, ch in zip(xs, word):
        d.text((x, by), ch, font=f, fill=255, anchor='ls', features=features,
               stroke_width=int(round(stroke * SS)), stroke_fill=255)
    a = np.asarray(im).astype(np.float32) / 255
    cols = np.nonzero(a.max(axis=0) > 0.25)[0]
    c = (cols.min() + cols.max() + 1) / 2
    shift = int(round((cx - x0) * SS - c))
    out = np.zeros((H * SS, W * SS), np.float32)
    src0 = max(0, -shift); dst0 = max(0, shift)
    n = min(W * SS - dst0, a.shape[1] - src0)
    out[:, dst0:dst0 + n] = a[:, src0:src0 + n]
    return out.reshape(H, SS, W, SS).mean(axis=(1, 3))


def iou(a, b, r=3):
    best = (0, 0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            bb = np.roll(np.roll(b, dy, 0), dx, 1)
            s = np.minimum(a, bb).sum() / max(1e-6, np.maximum(a, bb).sum())
            if s > best[0]: best = (s, dx, dy)
    return best


# settled frames, white text on dark backgrounds
SANS_REFS = [  # word, frame, crop box, baseline y, ink centre x
    ('winning', 142, (740, 470, 1170, 615)),
    ('and', 146, (840, 470, 1075, 615)),
    ('charging', 78, (730, 495, 1180, 615)),
    ('still', 69, (860, 495, 1080, 615)),
]


def ref_geom(n, box):
    m = ref_mask(n, box)
    cols = np.nonzero((m > 0.5).sum(0) >= 3)[0]
    cx = box[0] + (cols.min() + cols.max() + 1) / 2
    return m, cx


def score(path, axes, size, trk, refs, base_y=579.5, r=3, features=None, stroke=0.0):
    tot = []
    for word, n, box, m, cx in refs:
        rm = render_mask(path, axes, size, trk, word, box, base_y, cx, features, stroke)
        tot.append(iou(m, rm, r)[0])
    return float(np.mean(tot)), tot


def prep(refs):
    out = []
    for word, n, box in refs:
        m, cx = ref_geom(n, box)
        out.append((word, n, box, m, cx))
    return out
