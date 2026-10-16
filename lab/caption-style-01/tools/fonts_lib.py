"""Shared helpers for the font-matching study (render candidates with PIL/raqm)."""
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01'
CANDS = LAB + '/downloads/fontcands'
FR = LAB + '/ref/frames'
RAQM = ImageFont.Layout.RAQM


def font_path(family_dir, pick=None):
    fs = sorted(glob.glob(os.path.join(CANDS, family_dir, '*.ttf')))
    fs = [f for f in fs if 'Italic' not in os.path.basename(f)] or fs
    if pick:
        fs = [f for f in fs if pick in os.path.basename(f)] or fs
    return fs[0]


def load(path, size, axes=None):
    """axes: dict like {'wght': 700, 'opsz': 32}; unspecified axes keep defaults."""
    f = ImageFont.truetype(path, size=max(1, int(round(size))), layout_engine=RAQM)
    if axes:
        try:
            info = f.get_variation_axes()
        except Exception:
            info = None
        if info:
            names = [a['name'] if isinstance(a['name'], str) else a['name'].decode() for a in info]
            vals = []
            for a, nm in zip(info, names):
                key = {'Weight': 'wght', 'Optical size': 'opsz', 'Optical Size': 'opsz',
                       'Width': 'wdth', 'Slant': 'slnt'}.get(nm, nm[:4].lower())
                v = axes.get(key, a.get('default'))
                v = min(max(v, a['minimum']), a['maximum'])
                vals.append(v)
            f.set_variation_by_axes(vals)
    return f


def axes_info(path):
    f = ImageFont.truetype(path, 50, layout_engine=RAQM)
    try:
        return [(a['name'], a['minimum'], a.get('default'), a['maximum']) for a in f.get_variation_axes()]
    except Exception:
        return []


def render_word(font, text, tracking_px=0.0, color=255, pad=20, features=None):
    """Render text white on black; tracking applied per glyph (as AE/CSS letter-spacing).
    Returns an L image cropped to the ink bbox plus pad, and baseline y inside it."""
    asc, desc = font.getmetrics()
    if tracking_px == 0:
        w = int(font.getlength(text, features=features)) + 2 * pad + 400
        h = asc + desc + 2 * pad
        im = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(im)
        d.text((pad + 200, pad), text, font=font, fill=color, features=features)
    else:
        # per-character placement using advances incl. kerning (pairwise)
        w = int(font.getlength(text, features=features) + abs(tracking_px) * len(text)) + 2 * pad + 400
        h = asc + desc + 2 * pad
        im = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(im)
        x = pad + 200.0
        for i, ch in enumerate(text):
            d.text((x, pad), ch, font=font, fill=color, features=features)
            if i + 1 < len(text):
                pair = font.getlength(text[i:i + 2], features=features)
                nxt = font.getlength(text[i + 1], features=features)
                x += (pair - nxt) + tracking_px
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 10)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    base = pad + asc
    crop = im.crop((x0 - pad, y0 - pad, x1 + pad + 1, y1 + pad + 1))
    return crop, base - (y0 - pad)


def glyph_metrics(font, ch):
    l, t, r, b = font.getbbox(ch, anchor='ls')
    adv = font.getlength(ch)
    return dict(lsb=l, rsb=adv - r, w=r - l, top=-t, bottom=b, adv=adv)


def ref_crop(n, box):
    return Image.open(f'{FR}/f{n:03d}.png').convert('RGB').crop(box)


def label(img, text, h=34, size=24):
    W = img.width
    out = Image.new('RGB', (W, img.height + h), (24, 24, 24))
    out.paste(img.convert('RGB'), (0, h))
    d = ImageDraw.Draw(out)
    try:
        lf = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except Exception:
        lf = ImageFont.load_default()
    d.text((8, 4), text, fill=(255, 220, 0), font=lf)
    return out


def stack(imgs, bg=(0, 0, 0), gap=6, width=None):
    W = width or max(i.width for i in imgs)
    H = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new('RGB', (W, H), bg)
    y = 0
    for i in imgs:
        out.paste(i.convert('RGB'), (0, y))
        y += i.height + gap
    return out


def grid(imgs, cols, bg=(0, 0, 0), gap=6):
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    cw = max(i.width for i in imgs)
    rh = [max(i.height for i in r) for r in rows]
    out = Image.new('RGB', (cols * cw + gap * (cols - 1), sum(rh) + gap * (len(rows) - 1)), bg)
    y = 0
    for r, h in zip(rows, rh):
        for j, i in enumerate(r):
            out.paste(i.convert('RGB'), (j * (cw + gap), y))
        y += h + gap
    return out
