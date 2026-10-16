import numpy as np, sys
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from textmask import text_mask, crop
from cleanmask import drop_border
from words import WORDS, KEEP
CACHE = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'
def overlay(I, M):
    o = I.copy()
    edge = M ^ ndi.binary_erosion(M)
    o[edge] = [1, 0, 1]
    return o
def make(wid, rep, box, model):
    I = crop(rep, box)
    M, B, sc = text_mask(I, ring=5, normal=(model == 'normal'))
    x0, y0 = box[:2]
    if wid in KEEP:
        kx0, ky0, kx1, ky1 = KEEP[wid]
        K = np.zeros_like(M); K[ky0-y0:ky1-y0, kx0-x0:kx1-x0] = True
        lab, n = ndi.label(M)
        if n:
            inside = ndi.sum(K & M, lab, range(1, n+1)); tot = ndi.sum(M, lab, range(1, n+1))
            good = 1 + np.nonzero((inside / np.maximum(tot, 1) >= 0.6) & (tot >= 12))[0]
            M = np.isin(lab, good) & K
    else:
        M = drop_border(M, margin=7)
    lab, n = ndi.label(M); sizes = ndi.sum(M, lab, range(1, n+1))
    M = np.isin(lab, 1+np.nonzero(sizes >= 40)[0])
    return I, M
if __name__ == '__main__':
    S = sys.argv[1]; ids = sys.argv[2:]
    tiles = []
    for wid, txt, f0, f1, rep, box, model in WORDS:
        if ids and wid not in ids: continue
        I, M = make(wid, rep, box, model)
        np.save(f'{CACHE}/ref_{wid}.npy', M)
        vis = np.concatenate([I, overlay(I, M)], 1)
        im = Image.fromarray((np.clip(vis, 0, 1)*255).astype(np.uint8)); im.thumbnail((900, 300))
        ImageDraw.Draw(im).text((4, 4), f'{wid} {txt} f{rep}', fill=(255, 255, 0))
        tiles.append(im)
    cols = 2; tw = 900; th = max(t.height for t in tiles); rows = (len(tiles)+cols-1)//cols
    Sh = Image.new('RGB', (cols*tw, rows*th), (30, 30, 30))
    for k, t in enumerate(tiles): Sh.paste(t, ((k % cols)*tw, (k//cols)*th))
    Sh.save(f'{S}/refmasks.png'); print(Sh.size)
