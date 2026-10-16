import numpy as np, sys
from PIL import Image, ImageDraw
from textmask import text_mask, crop
from words import WORDS
S = sys.argv[1]
only = sys.argv[2:] 
tiles = []
for wid, txt, f0, f1, rep, box, model in WORDS:
    if only and wid not in only: continue
    I = crop(rep, box)
    M, B, sc = text_mask(I, normal=(model=='normal'))
    vis = np.concatenate([I, np.repeat(M[..., None], 3, -1).astype(np.float32)], 1)
    im = Image.fromarray((np.clip(vis, 0, 1)*255).astype(np.uint8))
    im.thumbnail((900, 260))
    d = ImageDraw.Draw(im); d.text((4, 4), f'{wid} {txt} f{rep}', fill=(255, 255, 0))
    tiles.append(im)
    np.save(f'/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/cache/raw_{wid}.npy', M)
cols = 2
tw = 900; th = max(t.height for t in tiles)
rows = (len(tiles)+cols-1)//cols
Sh = Image.new('RGB', (cols*tw, rows*th), (30, 30, 30))
for k, t in enumerate(tiles):
    Sh.paste(t, ((k % cols)*tw, (k//cols)*th))
Sh.save(f'{S}/repmasks.png'); print(Sh.size)
