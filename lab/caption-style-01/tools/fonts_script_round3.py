"""Round-3 script sheet: every reference script word at 1:1 vs the finalists rendered at the
reference x-height / baseline / centre (white on black)."""
import json
from fonts_lib import *
from fonts_script_fit import ref_x, fam_path

A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
WORDS = [  # word, frame, crop box, baseline y, x-height px, ink centre x
    ('Prime', 32, (560, 330, 1440, 740), 672, 115, 1003),
    ('hundred', 55, (600, 400, 1380, 670), 631.5, 83, 990),
    ('They', 96, (240, 300, 800, 660), 562, 80, 525),
    ('Win', 113, (760, 360, 1320, 700), 643, 82, 1036),
    ('Regardless', 152, (620, 410, 1340, 690), 600, 55, 972),
]
# size factor = font px per px of reference x-height, from the IoU fit on 'hundred' (fonts_script_fit2.py)
FINALISTS = [
    ('Pinyon Script (chosen)', 'pinyonscript', None, -3.0, 0.5, 216.4 / 83),
    ('Great Vibes ss02 caps', 'greatvibes', ['ss02'], 4.5, 0.0, 265.8 / 83),
    ('Imperial Script', 'imperialscript', None, 4.5, 0.5, 263.0 / 83),
]


def render(path, word, box, base, xh_px, cx, feats=None, shear=0.0, stroke=0.0, factor=2.6, SS=3):
    size = xh_px * factor
    W, H = box[2] - box[0], box[3] - box[1]
    f = load(path, size * SS)
    im = Image.new('L', (W * SS * 2, H * SS), 0); d = ImageDraw.Draw(im)
    by = (base - box[1]) * SS
    d.text((W * SS // 2, by), word, font=f, fill=255, anchor='ls', features=feats,
           stroke_width=int(round(stroke * SS)), stroke_fill=255)
    if shear:
        k = np.tan(np.radians(shear))
        im = im.transform(im.size, Image.AFFINE, (1, k, -k * by, 0, 1, 0), resample=Image.BICUBIC)
    a = np.asarray(im); cols = np.nonzero(a.max(0) > 60)[0]
    c = (cols.min() + cols.max()) / 2
    off = int(round((cx - box[0]) * SS - c))
    out = Image.new('L', (W * SS, H * SS), 0); out.paste(im, (off, 0))
    return out.resize((W, H), Image.LANCZOS), size


def main():
    cols = []
    for word, n, box, base, xh, cx in WORDS:
        tiles = [label(ref_crop(n, box), f'REF "{word}" f{n:03d}  x-height {xh}px', size=18, h=26)]
        for name, fam, feats, shear, stroke, factor in FINALISTS:
            im, size = render(fam_path(fam), word, box, base, xh, cx, feats, shear, stroke, factor)
            tiles.append(label(im, f'{name}  {size:.0f}px', size=18, h=26))
        cols.append(stack(tiles, gap=4))
    a = stack([cols[0], cols[2]], gap=8, bg=(60, 60, 60)) if False else None
    row1 = cols[:3]; row2 = cols[3:]
    def hcat(cs):
        sh = Image.new('RGB', (sum(c.width for c in cs) + 8 * (len(cs) - 1), max(c.height for c in cs)), (60, 60, 60)); x = 0
        for c in cs:
            sh.paste(c, (x, 0)); x += c.width + 8
        return sh
    hcat(row1).save(A + 'fonts_script_round3a.png'); hcat(row2).save(A + 'fonts_script_round3b.png')
    print('ok')


if __name__ == '__main__':
    main()
