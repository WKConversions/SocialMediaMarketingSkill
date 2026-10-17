#!/usr/bin/env python3
"""Same-scale check of the script class against the reference: sans ascender normalised to the
reference's 80 px, then 'hundred' (ref f055) / 'Win' (ref f113) next to our script words, for a few
stem_boost / stroke settings. Usage: python3 test_script_weight.py OUT.png"""
import copy, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_captions import Typesetter

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
style = json.load(open(f'{LAB}/styles/prime.json'))
H = 1920
sans_px = style['layouts']['talking_head']['sans_size_frac_of_height'] * H
variants = [
    ('v1: size 1.9, swash 2.6, stroke .0023, no boost', dict(size_rel_sans=1.9, stroke_em=0.0023, stem_boost=None), 2.6),
    ('critic: size 2.2, swash 2.9, stroke .005', dict(size_rel_sans=2.2, stroke_em=0.005, stem_boost=None), 2.9),
    ('v2: size 2.2, swash 2.9, stroke .0023 + stem_boost', {}, 2.9),
]
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)


def tile(img, label):
    out = Image.new('RGB', (img.width, img.height + 30), (20, 20, 20))
    out.paste(img, (0, 30))
    ImageDraw.Draw(out).text((6, 5), label, font=F, fill=(255, 210, 0))
    return out


rows = []
ts0 = Typesetter(style, f'{LAB}/styles')
asc = ts0.render('l', 'sans', sans_px)
asc_h = asc.ink[3] - asc.ink[1]
k = 80.0 / asc_h          # scale our pixels to the reference's (sans ascender = 80 px)
ref = Image.open(f'{LAB}/ref/frames/f055.png').convert('RGB').crop((560, 380, 1360, 720))
ref2 = Image.open(f'{LAB}/ref/frames/f113.png').convert('RGB').crop((560, 330, 1360, 720))
rows.append(tile(ref, 'REF f055 hundred (x-height 83 px)'))
rows.append(tile(ref2, 'REF f113 Win (W ~249 px = 3.1x sans ascender)'))
for label, over, swash in variants:
    stv = copy.deepcopy(style)
    c = stv['classes']['script']
    for kk, v in over.items():
        if v is None:
            c.pop(kk, None)
        else:
            c[kk] = v
    c['swash_capital']['height_x_ratio'] = swash
    ts = Typesetter(stv, f'{LAB}/styles')
    ims = []
    info = []
    for w in ['Editen', 'Willen', 'hundred']:
        r = ts.render(w, 'script', sans_px * c['size_rel_sans'])
        a = r.alpha[r.ink[1]:r.ink[3], r.ink[0]:r.ink[2]]
        im = Image.fromarray((a * 255).astype(np.uint8))
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
        ims.append(im)
        info.append(f'{w} {r.ink[3] - r.ink[1]}px={(r.ink[3] - r.ink[1]) / asc_h:.2f}x asc')
    wsum = sum(i.width for i in ims) + 20 * len(ims)
    hmax = max(i.height for i in ims)
    canvas = Image.new('L', (wsum, hmax), 0)
    x = 0
    for i in ims:
        canvas.paste(i, (x, hmax - i.height))
        x += i.width + 20
    rows.append(tile(canvas.convert('RGB'), f'{label} | ' + ', '.join(info)))
Wm = max(r.width for r in rows)
sheet = Image.new('RGB', (Wm, sum(r.height + 8 for r in rows)), (40, 40, 40))
y = 0
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height + 8
sheet.save(sys.argv[1])
print('sans ascender px', asc_h, 'scale to ref', round(k, 3))
