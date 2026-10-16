"""Hybrid test: Pinyon Script lowercase + enlarged swash capital (Pinyon itself, or Imperial Script / Great Vibes caps)."""
from fonts_lib import *
from fonts_script_fit import fam_path
from fonts_script_round3 import WORDS

A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
PIN = fam_path('pinyonscript')
PIN_K = 216.4 / 83          # Pinyon px per px of x-height
CAP_TARGET = 2.5            # reference swash-cap height / x-height


def render_hybrid(word, box, base, xh, cx, cap_font=None, cap_feats=None, cap_scale=None, SS=3, overlap=0.15, shear=-3.0):
    W, H = box[2] - box[0], box[3] - box[1]
    im = Image.new('L', (W * SS * 2, H * SS), 0); d = ImageDraw.Draw(im)
    by = (base - box[1]) * SS
    low = load(PIN, xh * PIN_K * SS)
    first, rest = word[0], word[1:]
    if first.isupper():
        cf = cap_font or PIN
        f0 = load(cf, 400)
        x0, top, x1, bottom = ink(f0, first, cap_feats)
        size = 400 * (CAP_TARGET * xh * SS) / top if cap_scale is None else xh * PIN_K * SS * cap_scale
        fc = load(cf, size)
        x0, top, x1, bottom = ink(fc, first, cap_feats)
        pen = W * SS * 0.5
        d.text((pen, by), first, font=fc, fill=255, anchor='ls', features=cap_feats, stroke_width=int(0.5 * SS), stroke_fill=255)
        # advance-based placement (keeps designed overhangs such as the T bar over 'he');
        # the capital's advance is taken at the size it is drawn at
        pen2 = pen + fc.getlength(first, features=cap_feats)
        d.text((pen2, by), rest, font=low, fill=255, anchor='ls', stroke_width=int(0.5 * SS), stroke_fill=255)
    else:
        d.text((W * SS * 0.5, by), word, font=low, fill=255, anchor='ls', stroke_width=int(0.5 * SS), stroke_fill=255)
    if shear:
        k = np.tan(np.radians(shear))
        im = im.transform(im.size, Image.AFFINE, (1, k, -k * by, 0, 1, 0), resample=Image.BICUBIC)
    a = np.asarray(im); cols = np.nonzero(a.max(0) > 60)[0]
    c = (cols.min() + cols.max()) / 2
    out = Image.new('L', (W * SS, H * SS), 0); out.paste(im, (int(round((cx - box[0]) * SS - c)), 0))
    return out.resize((W, H), Image.LANCZOS)


if __name__ == '__main__':
    variants = [('Pinyon, capital at native size', None, None, 1.0),
                ('Pinyon, capital enlarged to 2.5x x-height', None, None, None),
                ('Pinyon + Imperial Script capital (2.5x x-height)', fam_path('imperialscript'), None, None),
                ('Pinyon + Great Vibes ss03 capital (2.5x x-height)', fam_path('greatvibes'), ['ss03'], None)]
    cols = []
    for word, n, box, base, xh, cx in WORDS:
        if word == 'hundred':
            continue
        tiles = [label(ref_crop(n, box), f'REF "{word}" f{n:03d}', size=18, h=26)]
        for name, cf, feats, sc in variants:
            tiles.append(label(render_hybrid(word, box, base, xh, cx, cf, feats, sc), name, size=18, h=26))
        cols.append(stack(tiles, gap=4))
    for i, cs in enumerate((cols[:2], cols[2:])):
        sh = Image.new('RGB', (sum(c.width for c in cs) + 8, max(c.height for c in cs)), (60, 60, 60)); x = 0
        for c in cs:
            sh.paste(c, (x, 0)); x += c.width + 8
        sh.save(A + f'fonts_script_round3_caps_hybrid_{"ab"[i]}.png')
    print('ok')
