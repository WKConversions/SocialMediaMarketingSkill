import numpy as np, sys, json
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from textmask import text_mask, crop
from words import WORDS
CACHE = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache'

def drop_border(M, margin=9):
    lab, n = ndi.label(M)
    if n == 0: return M
    h, w = M.shape
    edge = np.zeros_like(M); edge[:margin] = edge[-margin:] = True; edge[:, :margin] = edge[:, -margin:] = True
    bad = np.unique(lab[edge & M])
    keep = ~np.isin(lab, bad) & M
    return keep

def build(wid, frames_used, box, model, core=None, persist=0.5):
    Ms = []
    for n in frames_used:
        M, B, sc = text_mask(crop(n, box), normal=(model == 'normal'))
        Ms.append(drop_border(M))
    P = np.mean(Ms, 0)
    return P, Ms

if __name__ == '__main__':
    S = sys.argv[1]; ids = sys.argv[2:]
    tiles = []
    for wid, txt, f0, f1, rep, box, model in WORDS:
        if ids and wid not in ids: continue
        fr = list(range(f0, f1+1))
        P, Ms = build(wid, fr, box, model)
        np.save(f'{CACHE}/P_{wid}.npy', P)
        I = crop(rep, box)
        vis = np.concatenate([I, np.repeat(P[..., None], 3, -1)], 1)
        im = Image.fromarray((np.clip(vis, 0, 1)*255).astype(np.uint8)); im.thumbnail((900, 260))
        ImageDraw.Draw(im).text((4, 4), f'{wid} {txt}', fill=(255, 255, 0))
        tiles.append(im)
    cols = 2; tw = 900; th = max(t.height for t in tiles); rows = (len(tiles)+cols-1)//cols
    Sh = Image.new('RGB', (cols*tw, rows*th), (30, 30, 30))
    for k, t in enumerate(tiles): Sh.paste(t, ((k % cols)*tw, (k//cols)*th))
    Sh.save(f'{S}/persist.png'); print(Sh.size)
