"""Script comparison sheets: reference crop on top, candidates (white on black) below."""
import json, sys
from fonts_fit import *
from fonts_script_fit import BOX, BASE
A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'


def hundred_tiles(names, fit, zoom=0.75, ref_frame=55):
    m = ref_mask(ref_frame, BOX, 0.3, 0.6)
    cols = np.nonzero((m > 0.5).sum(0) >= 2)[0]
    cx = BOX[0] + (cols.min() + cols.max() + 1) / 2
    ref = ref_crop(ref_frame, BOX)
    W, H = int(ref.width * zoom), int(ref.height * zoom)
    tiles = [label(ref.resize((W, H), Image.LANCZOS), f'REFERENCE "hundred" f{ref_frame:03d}', size=20, h=28)]
    for nm in names:
        v = fit[nm]
        rm = render_mask(v['path'], v.get('axes'), v['size'], v['track_em'], 'hundred', BOX, BASE, cx)
        s, dx, dy = iou(m, rm, 4); rm = np.roll(np.roll(rm, dy, 0), dx, 1)
        im = Image.fromarray((rm * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
        tiles.append(label(im, f'{nm}  IoU {s:.3f}  {v["size"]:.0f}px', size=20, h=28))
    return tiles


if __name__ == '__main__':
    fit = json.load(open(A + 'cache_script_fit.json'))
    names = [k for k, v in sorted(fit.items(), key=lambda kv: -kv[1]['iou'])]
    tiles = hundred_tiles(names, fit, 0.6)
    ref = tiles[0]
    per = 11
    chunks = [names[i:i + per] for i in range(0, len(names), per)]
    cols = []
    k = 1
    for ch in chunks:
        cols.append(stack([ref] + tiles[k:k + len(ch)], gap=3)); k += len(ch)
    # 2 sheets of 2 columns
    for si in range(0, len(cols), 2):
        cs = cols[si:si + 2]
        W = sum(c.width for c in cs) + 8 * (len(cs) - 1)
        sh = Image.new('RGB', (W, max(c.height for c in cs)), (60, 60, 60)); x = 0
        for c in cs:
            sh.paste(c, (x, 0)); x += c.width + 8
        fn = A + f'fonts_script_round1_hundred_{si // 2 + 1}.png'
        sh.save(fn); print(fn, sh.size)
