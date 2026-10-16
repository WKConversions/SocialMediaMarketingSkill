"""Estimate global zoom (scale about a centre) + shift between two frames, excluding a text box."""
import numpy as np, sys
from PIL import Image
from common import frames, W, H
a = frames()
def lum(n, ds=2):
    im = Image.fromarray(np.asarray(a[n-1])).convert('L').resize((W//ds, H//ds), Image.BILINEAR)
    return np.asarray(im).astype(np.float32)/255
def warp(img, s, dx, dy, cx, cy):
    # output(x) = img(cx + (x-cx)/s - dx/s ...) : scale about (cx,cy) by s then shift by dx,dy
    im = Image.fromarray(img)
    inv = 1/s
    # PIL affine: input = A*output + b
    A = (inv, 0, cx - inv*(cx+dx), 0, inv, cy - inv*(cy+dy))
    return np.asarray(im.transform(img.shape[::-1], Image.AFFINE, A, resample=Image.BILINEAR))
def grad(l):
    gx = np.zeros_like(l); gy = np.zeros_like(l)
    gx[:,1:-1] = l[:,2:]-l[:,:-2]; gy[1:-1] = l[2:]-l[:-2]
    return gx, gy
def score(la, lb_w, mask):
    ga = grad(la); gb = grad(lb_w)
    num = (ga[0]*gb[0]+ga[1]*gb[1])[mask].sum()
    den = np.sqrt(((ga[0]**2+ga[1]**2)[mask].sum())*((gb[0]**2+gb[1]**2)[mask].sum()))
    return num/den
def est(na, nb, textbox, ds=2, cx=None, cy=None):
    la, lb = lum(na, ds), lum(nb, ds)
    h, w = la.shape
    cx = (W/2/ds) if cx is None else cx/ds; cy = (H/2/ds) if cy is None else cy/ds
    mask = np.ones((h, w), bool)
    x0,y0,x1,y1 = [v//ds for v in textbox]
    mask[y0:y1, x0:x1] = False
    mask[:20] = mask[-20:] = False; mask[:, :20] = mask[:, -20:] = False
    best = (-1,)
    for s in np.arange(0.94, 1.06, 0.005):
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                sc = score(la, warp(lb, s, dx, dy, cx, cy), mask)
                if sc > best[0]: best = (sc, s, dx, dy)
    # refine
    sc0, s0, dx0, dy0 = best
    for s in np.arange(s0-0.005, s0+0.0051, 0.001):
        for dx in np.arange(dx0-2, dx0+2.1, 0.5):
            for dy in np.arange(dy0-2, dy0+2.1, 0.5):
                sc = score(la, warp(lb, s, dx, dy, cx, cy), mask)
                if sc > best[0]: best = (sc, s, dx, dy)
    return best
if __name__ == '__main__':
    na, nb = int(sys.argv[1]), int(sys.argv[2])
    tb = tuple(int(v) for v in sys.argv[3].split(','))
    sc, s, dx, dy = est(na, nb, tb)
    print(f'f{na}->f{nb}: scale(b->a)={s:.4f} shift=({dx*2:.1f},{dy*2:.1f})px score={sc:.3f}')
