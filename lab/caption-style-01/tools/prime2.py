import numpy as np, json, sys
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from prime import F, plate, CACHE
from common import LAB
def fitC(I, B):
    Cs = []
    for c in range(3):
        vs = np.linspace(0, 1, 201)
        er = [((I[:, c] - np.abs(v - B[:, c]))**2).mean() for v in vs]
        Cs.append(vs[int(np.argmin(er))])
    return np.array(Cs)
out = {}
for name, n, src in [('Prime', 28, 27), ('at my', 27, 28)]:
    I = F(n); P = plate(src, n); M = np.load(f'{CACHE}/prime_mask_{n}.npy')
    Mi = ndi.binary_erosion(M, iterations=2)
    ys, xs = np.nonzero(Mi)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bins = {}
    for i in range(4):
        for j in range(2):
            sel = (xs >= x0 + i*(x1-x0)/4) & (xs <= x0 + (i+1)*(x1-x0)/4) & (ys >= y0 + j*(y1-y0)/2) & (ys <= y0 + (j+1)*(y1-y0)/2)
            if sel.sum() < 200: continue
            C = fitC(I[ys[sel], xs[sel]], P[ys[sel], xs[sel]])
            bins[f'x{i}_y{j}'] = dict(C=C.round(3).tolist(), n=int(sel.sum()))
    # bright-bg vs dark-bg split
    Bl = P[ys, xs].mean(1)
    for lab_, sel in [('bg_dark(<0.3)', Bl < 0.3), ('bg_bright(>0.5)', Bl > 0.5)]:
        if sel.sum() > 200:
            C = fitC(I[ys[sel], xs[sel]], P[ys[sel], xs[sel]])
            bins[lab_] = dict(C=C.round(3).tolist(), n=int(sel.sum()))
    Call = fitC(I[ys, xs], P[ys, xs])
    out[name] = dict(C_all=Call.round(3).tolist(), bins=bins)
    print(name, Call, json.dumps(bins))
    # render model
    Ms = ndi.gaussian_filter(M.astype(np.float32), 0.7)[..., None]
    model = P*(1-Ms) + np.abs(Call - P)*Ms
    err = np.abs(model - I).mean(-1, keepdims=True).repeat(3, -1)*3
    sl = (slice(370, 700), slice(580, 1420)) if name == 'Prime' else (slice(490, 620), slice(810, 1110))
    tiles = [I[sl], P[sl], model[sl], np.clip(err[sl], 0, 1)]
    vis = np.concatenate(tiles, 0)
    im = Image.fromarray((np.clip(vis, 0, 1)*255).astype(np.uint8))
    d = ImageDraw.Draw(im); h = tiles[0].shape[0]
    for k, t in enumerate([f'actual f{n:03d}', f'clean plate (f{src:03d} warped)', f'model |C - plate|, C=RGB{tuple(int(v) for v in np.round(Call*255))}', 'abs error x3']):
        d.rectangle([0, k*h, 8*len(t)+8, k*h+14], fill=(0, 0, 0)); d.text((4, k*h+2), t, fill=(255, 255, 0))
    im.save(f'{LAB}/analysis/spec_prime_{name.replace(" ", "_")}.png')
json.dump(out, open(f'{CACHE}/prime_bins.json', 'w'), indent=1)
