"""Grade for the Röntgen edit: the client filmed in a dark room under red light, like a photographer's darkroom
(where X-ray plates were developed). The grade keeps that red, lifts the face out of the dark, deepens the blacks
with the hint of blue already in the shadows, and adds a vignette and fine grain. 'now' (the closing part) is a touch
brighter and cleaner. Float RGB 0..1 in and out.
Usage (test): python3 grade_darkroom.py IN.png OUT.png history|now
"""
import sys

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_maps = {}


def _vignette(h, w):
    if (h, w) not in _maps:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        u, v = (xx - w / 2) / (w / 2), (yy - h * 0.46) / (h / 2)
        face = np.exp(-((u / 0.50) ** 2 + ((v - 0.02) / 0.34) ** 2))
        _maps[(h, w)] = ((1 - 0.38 * (u ** 2 * 0.8 + v ** 2 * 0.5))[..., None], face[..., None])
    return _maps[(h, w)]


def grade_darkroom(x, era="history", rng=None):
    h, w = x.shape[:2]
    vig, face = _vignette(h, w)
    now = era == "now"
    x = x * ((1.30 if now else 1.22) + face * 0.22)                         # lift the face out of the dark
    x = np.where(x > 0.75, 0.75 + (1 - np.exp(-(x - 0.75) / 0.25)) * 0.25 * 0.9, x)
    lum = (x @ LUMA)[..., None]
    x = lum + (x - lum) * (0.86 if now else 0.82)                           # the red is very saturated: calm it a little
    x = np.clip(x, 0, 1)
    x = x * x * (3 - 2 * x) * 0.35 + x * 0.65                               # contrast
    lum = (x @ LUMA)[..., None]
    x = x + ((1 - lum) ** 4) * np.array([-0.012, -0.004, 0.030], np.float32)  # blue in the deepest shadows
    x = np.clip(x - 0.015, 0, 1) / 0.985                                     # true blacks
    x = x * vig
    if rng is not None:
        g = rng.normal(0, 0.012, (h // 2, w // 2)).astype(np.float32)
        x = x + np.repeat(np.repeat(g, 2, 0), 2, 1)[:h, :w, None]
    return np.clip(x, 0, 1)


if __name__ == "__main__":
    from PIL import Image
    src = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.float32) / 255
    out = grade_darkroom(src, sys.argv[3], np.random.default_rng(1))
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(sys.argv[2])
