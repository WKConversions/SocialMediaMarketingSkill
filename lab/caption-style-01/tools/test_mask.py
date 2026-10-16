import sys, numpy as np
from PIL import Image
from textmask import text_mask, crop
S = sys.argv[1]
tests = [(19, (700,400,1220,680)), (123,(640,380,1300,700)), (51,(500,360,1450,720)), (9,(700,420,1250,660)), (101,(700,420,1250,660)), (134,(700,420,1250,660)), (30,(300,180,1650,900))]
rows=[]
for n, box in tests:
    I = crop(n, box)
    M, B, sc = text_mask(I)
    vis = np.concatenate([I, np.repeat(M[...,None],3,-1).astype(np.float32), B], 1)
    im = Image.fromarray((np.clip(vis,0,1)*255).astype(np.uint8))
    im.thumbnail((1500, 400))
    im.save(f'{S}/m_{n}.png')
    ys, xs = np.nonzero(M)
    print(n, 'bbox', xs.min()+box[0], ys.min()+box[1], xs.max()+box[0], ys.max()+box[1], 'px', M.sum())
