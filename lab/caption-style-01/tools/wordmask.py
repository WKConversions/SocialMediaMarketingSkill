import numpy as np, sys
from PIL import Image
from textmask import text_mask, crop

def label(m):
    """simple connected components (4-neigh) via union-find on runs -> use iterative flood with numpy"""
    h, w = m.shape
    lab = np.zeros((h, w), np.int32)
    cur = 0
    ys, xs = np.nonzero(m)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]: continue
        cur += 1
        stack = [(y0, x0)]; lab[y0, x0] = cur
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur; stack.append((yy, xx))
    return lab, cur

def word_score(nums, box, normal=False):
    acc = None
    for n in nums:
        I = crop(n, box)
        M, B, sc = text_mask(I, normal=normal)
        acc = sc if acc is None else acc + sc
    return acc / len(nums)

if __name__ == '__main__':
    S = sys.argv[1]
    for name, nums, box in [('even', range(9,17), (250,250,1670,850)), ('know', range(122,126), (250,250,1670,850))]:
        sc = word_score(list(nums), box)
        im = Image.fromarray((np.clip(sc*0.5+0.5,0,1)*255).astype(np.uint8)); im.thumbnail((1000,500)); im.save(f'{S}/ws_{name}.png')
