"""Colour grade for the Montgolfier 1783 -> WKC story: blue and gold, the colours of the 1783 Versailles balloon.

talking head, era 'history' (the 1783 story, c1-c2): warm gold highlights and skin, deep blue shadows, a little
    desaturated, soft highlight roll-off (tames the blown window), the flat ceiling at the top darkened like a
    graduated filter so the eye goes to the face, warm halation on the highlights, fine grain.
talking head, era 'now' (WKC, c3-c4): the same family, cleaner and brighter: natural skin, more contrast, less
    grain, a touch more saturation (conversion: clear, trustworthy, the face reads first).
archival prints: keep the print's own paper and ink (they're engravings, not film), warmed toward parchment,
    gently contrasted, with a soft vignette; hand-coloured prints keep their colour, slightly muted.
All functions take and return float32 RGB in 0..1, shape (H, W, 3).
Usage (test): python3 grade.py IN.png OUT.png history|now
"""
import sys

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_cache = {}


def _maps(h, w):
    if (h, w) not in _cache:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        u, v = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
        vig = 1 - 0.30 * (u ** 2 * 0.9 + v ** 2 * 0.50)                               # oval vignette
        top = np.clip((0.44 - yy / h) / 0.40, 0, 1) ** 1.1                              # graduated filter on the ceiling
        face = np.exp(-((u / 0.55) ** 2 + ((v - 0.12) / 0.32) ** 2))                    # soft light on the face (centre)
        _cache[(h, w)] = (vig[..., None], top[..., None], face[..., None])
    return _cache[(h, w)]


def _blur(x, r):
    """Cheap separable box blur (3 passes ~ gaussian) on a downscaled copy."""
    from PIL import Image, ImageFilter
    h, w = x.shape[:2]
    small = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((w // 4, h // 4), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(r / 4))
    return np.asarray(small.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255


def grade_talking(x, era="history", rng=None):
    hist = era == "history"
    h, w = x.shape[:2]
    vig, top, face = _maps(h, w)
    # 1. exposure and highlight roll-off: lift the face a touch, compress the window
    x = x * ((1.02 if hist else 1.06) + face * (0.10 if hist else 0.10))
    x = np.where(x > 0.72, 0.72 + (1 - np.exp(-(x - 0.72) / 0.28)) * 0.28 * 0.92, x)
    lum = x @ LUMA
    # 2. saturation, keeping skin (warm hues) a little richer than the rest
    warm = np.clip((x[..., 0] - x[..., 2]) * 4, 0, 1)[..., None]
    sat = (0.72 if hist else 0.92) + warm * (0.14 if hist else 0.08)
    x = lum[..., None] + (x - lum[..., None]) * sat
    # 3. contrast: soft S-curve
    k = 0.38 if hist else 0.45
    x = np.clip(x, 0, 1)
    x = x * (1 - k) + (x * x * (3 - 2 * x)) * k
    # 4. split tone: blue shadows, gold highlights
    lum = np.clip(x @ LUMA, 0, 1)[..., None]
    sh, hi = (1 - lum) ** 2.2, lum ** 1.8
    if hist:
        x = x + sh * np.array([-0.045, -0.005, 0.060], np.float32) + hi * np.array([0.080, 0.035, -0.070], np.float32)
    else:
        x = x + sh * np.array([-0.030, -0.002, 0.040], np.float32) + hi * np.array([0.040, 0.016, -0.032], np.float32)
    # 5. darken the ceiling (graduated filter, slightly blue like an evening sky) and vignette
    x = x * (1 - top * (0.62 if hist else 0.50)) + top * np.array([0.0, 0.006, 0.030], np.float32)
    x = x * vig
    # 6. warm halation on the brightest areas (window), only a whisper
    glow = np.clip(x @ LUMA - 0.70, 0, 1)[..., None] * np.array([1.0, 0.55, 0.25], np.float32)
    x = x + _blur(glow, 40) * (0.18 if hist else 0.12)
    # 7. fine grain (luma)
    if rng is not None:
        g = rng.normal(0, 0.014 if hist else 0.008, (h // 2, w // 2)).astype(np.float32)
        x = x + np.repeat(np.repeat(g, 2, 0), 2, 1)[:h, :w, None]
    return np.clip(x, 0, 1)


def grade_print(x, rng=None, colour=False):
    """Engravings and hand-coloured prints: parchment warmth, gentle contrast, vignette, a little paper texture."""
    h, w = x.shape[:2]
    vig, _, _ = _maps(h, w)
    lum = x @ LUMA
    if colour:
        x = lum[..., None] + (x - lum[..., None]) * 0.85
    else:
        x = np.repeat(lum[..., None], 3, axis=2)
    x = np.clip((x - 0.5) * 1.12 + 0.5, 0, 1)
    tone = np.array([1.03, 0.985, 0.90], np.float32)                                  # parchment
    x = x * tone + (1 - x) * np.array([0.0, 0.005, 0.02], np.float32) * 0.6            # blue-ish ink in the darks
    x = x * (vig * 0.92 + 0.08 * vig ** 3)
    if rng is not None:
        x = x + rng.normal(0, 0.010, (h, w, 1)).astype(np.float32)
    return np.clip(x, 0, 1)


if __name__ == "__main__":
    from PIL import Image
    src = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.float32) / 255
    out = grade_talking(src, sys.argv[3], np.random.default_rng(1)) if sys.argv[3] in ("history", "now") else grade_print(src)
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(sys.argv[2])
