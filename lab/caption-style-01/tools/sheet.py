import sys, numpy as np
from PIL import Image, ImageDraw
from common import frames, LAB
a = frames()
def sheet(nums, out, cols=6, tw=320, crop=None):
    th = int(tw* (crop[3]-crop[1])/(crop[2]-crop[0])) if crop else int(tw*1078/1916)
    rows = (len(nums)+cols-1)//cols
    S = Image.new('RGB', (cols*tw, rows*th), (40,40,40))
    d = ImageDraw.Draw(S)
    for k,n in enumerate(nums):
        im = Image.fromarray(np.asarray(a[n-1]))
        if crop: im = im.crop(crop)
        im = im.resize((tw,th), Image.LANCZOS)
        x,y = (k%cols)*tw, (k//cols)*th
        S.paste(im,(x,y))
        d.rectangle([x,y,x+38,y+14], fill=(0,0,0)); d.text((x+2,y+1), f'f{n:03d}', fill=(255,255,0))
    S.save(out)
if __name__ == '__main__':
    a0, a1 = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3]
    cols = int(sys.argv[4]) if len(sys.argv)>4 else 6
    tw = int(sys.argv[5]) if len(sys.argv)>5 else 320
    crop = tuple(int(v) for v in sys.argv[6].split(',')) if len(sys.argv)>6 else None
    sheet(list(range(a0,a1+1)), out, cols, tw, crop)
