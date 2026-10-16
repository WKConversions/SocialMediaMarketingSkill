"""Text mask extraction for white-text-in-Difference captions.
Model: I = B outside text, I = 1 - B inside text (B = background).
Iterative: estimate smooth B from un-differenced image U, reclassify."""
import numpy as np
from common import frames

def boxblur(x, r, axis):
    if r < 1: return x
    k = 2*r+1
    pad = [(0,0)]*x.ndim; pad[axis] = (r+1, r)
    c = np.cumsum(np.pad(x, pad, mode='edge'), axis=axis)
    sl_hi = [slice(None)]*x.ndim; sl_lo = [slice(None)]*x.ndim
    sl_hi[axis] = slice(k, None); sl_lo[axis] = slice(0, -k)
    return (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / k

def gblur(x, sigma):
    r = max(1, int(round(sigma*0.93)))  # 3 box passes ~ gaussian
    y = x
    for ax in (0, 1):
        for _ in range(3):
            y = boxblur(y, r, ax)
    return y

def nconv(val, w, sigma):
    if val.ndim == 3:
        num = np.stack([gblur(val[..., c]*w, sigma) for c in range(val.shape[-1])], -1)
        den = gblur(w, sigma)[..., None]
    else:
        num = gblur(val*w, sigma); den = gblur(w, sigma)
    return num / np.maximum(den, 1e-6)

def text_mask(I, ring=6, sigmas=(40, 16, 8, 5), iters=2, normal=False):
    """I: HxWx3 float crop whose border ring is background only.
    returns (mask bool, B estimate, score map)"""
    h, w, _ = I.shape
    W = np.zeros((h, w), np.float32)
    W[:ring] = W[-ring:] = 1; W[:, :ring] = W[:, -ring:] = 1
    B = nconv(I, W, sigmas[0])
    M = np.zeros((h, w), bool)
    for s in sigmas:
        for _ in range(iters):
            T = np.ones_like(I) if normal else 1 - B
            dT = ((I - T)**2).sum(-1); dB = ((I - B)**2).sum(-1)
            M = dT < dB
            M[:ring] = M[-ring:] = False; M[:, :ring] = M[:, -ring:] = False
            if normal:
                U = I; wgt = (~M).astype(np.float32)
            else:
                U = np.where(M[..., None], 1 - I, I); wgt = np.ones((h, w), np.float32)
            # downweight edge pixels (antialiasing) for the B estimate
            conf = np.abs(dT - dB) / (dT + dB + 1e-4)
            B = nconv(U, wgt*conf, s)
    T = np.ones_like(I) if normal else 1 - B
    dT = ((I - T)**2).sum(-1); dB = ((I - B)**2).sum(-1)
    return M, B, (dB - dT) / (dT + dB + 1e-4)

def crop(n, box):
    x0, y0, x1, y1 = box
    return frames()[n-1, y0:y1, x0:x1].astype(np.float32) / 255
