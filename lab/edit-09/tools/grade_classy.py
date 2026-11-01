"""Classy grade for the Wright brothers edit, the client's direction: stay near the original colours, sharpen a bit,
a little more contrast for a slightly low-tone vibe (the footage is already low: careful).

1. Original colours as the base: no white-balance shift, no recolouring (a recolour made the skin look grey).
2. Light sharpening: the clips are compressed and soft; a fine unsharp mask brings back eyes, hair, suit.
3. A little contrast and slightly lower mid-tones (evening vibe); blacks a touch deeper, never crushed.
4. Only the most saturated reds calmed (~20%), so the cheeks don't glow red while the warmth stays.
5. A whisper of film grain and a soft vignette.
Float RGB 0..1 in and out; sharpening runs on the 8-bit frame first (sharpen()).
Usage (test): python3 grade_classy.py IN.png OUT.png
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_maps = {}


def sharpen(frame_u8):
    """Fine unsharp mask on the 8-bit frame (radius 1.4 px, 75%, threshold 2): detail back, no halos."""
    return np.asarray(Image.fromarray(frame_u8).filter(ImageFilter.UnsharpMask(radius=1.4, percent=75, threshold=2)))


def _vignette(h, w):
    if (h, w) not in _maps:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        u, v = (xx - w / 2) / (w / 2), (yy - h * 0.45) / (h / 2)
        _maps[(h, w)] = (1 - 0.16 * np.clip(u ** 2 * 0.8 + v ** 2 * 0.55, 0, 1.4))[..., None].astype(np.float32)
    return _maps[(h, w)]


def calm_strong_reds(x, amount=0.20):
    """Desaturate only clearly red-dominant pixels (by up to `amount`) toward their own brightness."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    redness = np.clip(((r - np.maximum(g, b)) / (r + 1e-4) - 0.35) / 0.5, 0, 1)
    lum = (x @ LUMA)[..., None]
    return x - (redness * amount)[..., None] * (x - lum)


def grade_classy(x, era="history", rng=None):
    h, w = x.shape[:2]
    x = calm_strong_reds(np.clip(x, 0, 1))
    x = np.clip(x, 0, 1) ** 1.04                                                    # slightly lower mid-tones
    x = x * x * (3 - 2 * x) * 0.22 + x * 0.78                                       # a little contrast
    x = x * _vignette(h, w)
    if rng is not None:
        gr = rng.normal(0, 0.007, (h // 2, w // 2)).astype(np.float32)
        x = x + np.repeat(np.repeat(gr, 2, 0), 2, 1)[:h, :w, None]
    return np.clip(x, 0, 1)


if __name__ == "__main__":
    src = sharpen(np.asarray(Image.open(sys.argv[1]).convert("RGB")))
    out = grade_classy(src.astype(np.float32) / 255, rng=np.random.default_rng(1))
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(sys.argv[2])
