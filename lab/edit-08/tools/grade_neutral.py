"""Neutral colour from red-light footage (client: "less red or near no red, normal colours, but my facial expression
and cheeks stay popped out as now").

Under a pure red light nearly all the picture detail is in the red channel (green and blue are dark and noisy). So the
red channel is used as the brightness (it holds the face's shape: cheeks, expression, shadows) and that brightness is
mapped onto natural colours: neutral dark tones in the shadows, skin tones in the mids, warm light in the highlights
(a gradient map). The contrast that makes the face pop is kept, the red cast is gone. Float RGB 0..1 in and out.
Usage (test): python3 grade_neutral.py IN.png OUT.png
"""
import sys

import numpy as np

# brightness -> colour stops: dark neutral, shadow, skin mid, skin light, highlight
STOPS = np.array([0.00, 0.22, 0.52, 0.78, 1.00], np.float32)
COLOURS = np.array([[0.022, 0.024, 0.030],
                    [0.150, 0.110, 0.095],
                    [0.640, 0.410, 0.310],
                    [0.900, 0.660, 0.530],
                    [1.000, 0.910, 0.830]], np.float32)
LUT = np.stack([np.interp(np.linspace(0, 1, 1024), STOPS, COLOURS[:, c]) for c in range(3)], axis=1).astype(np.float32)


_maps = {}


def _falloff(h, w):
    """Soft darkening away from the face (the bright red wall would otherwise turn into a big pale area)."""
    if (h, w) not in _maps:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        u, v = (xx - w / 2) / (w / 2), (yy - h * 0.50) / (h / 2)
        vig = 1 - 0.45 * np.clip(u ** 2 * 0.7 + v ** 2 * 0.6, 0, 1)
        top = 1 - 0.45 * np.clip((0.40 - yy / h) / 0.40, 0, 1) ** 1.2
        _maps[(h, w)] = (vig * top).astype(np.float32)
    return _maps[(h, w)]


def grade_neutral(x, era="history", rng=None, black=0.03, white=0.92, gamma=1.35):
    lum = 0.88 * x[..., 0] + 0.12 * x[..., 1]                      # the red channel carries the detail
    lum = np.clip((lum - black) / (white - black), 0, 1) ** gamma   # deeper contrast: the face pops, the wall drops back
    lum = lum * _falloff(*lum.shape)
    return LUT[(np.clip(lum, 0, 1) * 1023).astype(np.int32)]


if __name__ == "__main__":
    from PIL import Image
    src = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.float32) / 255
    Image.fromarray((grade_neutral(src) * 255 + 0.5).astype(np.uint8)).save(sys.argv[2])
