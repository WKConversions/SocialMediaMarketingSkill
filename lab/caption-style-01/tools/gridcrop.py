import sys, numpy as np
from PIL import Image, ImageDraw
from common import frames
a = frames()
def gridcrop(n, box, out, step=50, scale=1.0):
    x0,y0,x1,y1 = box
    im = Image.fromarray(np.asarray(a[n-1, y0:y1, x0:x1]))
    if scale != 1: im = im.resize((int((x1-x0)*scale), int((y1-y0)*scale)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for x in range((x0//step+1)*step, x1, step):
        X = (x-x0)*scale; d.line([(X,0),(X,im.height)], fill=(255,255,0) if x%100==0 else (0,160,255), width=1)
        if x % 100 == 0: d.text((X+2, 2), str(x), fill=(255,255,0))
    for y in range((y0//step+1)*step, y1, step):
        Y = (y-y0)*scale; d.line([(0,Y),(im.width,Y)], fill=(255,255,0) if y%100==0 else (0,160,255), width=1)
        if y % 100 == 0: d.text((2, Y+2), str(y), fill=(255,255,0))
    d.text((im.width-40, 2), f'f{n}', fill=(255,0,255))
    im.save(out)
if __name__ == '__main__':
    n = int(sys.argv[1]); box = tuple(int(v) for v in sys.argv[2].split(',')); out = sys.argv[3]
    sc = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    gridcrop(n, box, out, scale=sc)
