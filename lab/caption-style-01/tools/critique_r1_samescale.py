"""Same typographic scale: reference crops vs ours, ours scaled so the sans ascender (k/l/d) matches the reference's 80 px."""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
S = Path(sys.argv[1]); LAB = Path('/home/user/SocialMediaMarketingSkill/lab/caption-style-01')
rep = json.load(open(LAB / 'out/user_v1.render.json'))
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
def lab(im, t):
    d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 26], fill=(20, 20, 20)); d.text((6, 3), t, font=F, fill=(255, 210, 0)); return im
K = 80 / 96.0   # ours sans ascender 96 px -> ref 80 px
def ref(n, box=(358, 330, 1558, 750)):
    return lab(Image.open(LAB / f'ref/frames/f{n:03d}.png').convert('RGB').crop(box), f'REF f{n:03d}')
def ours(n, box=(0, 1150, 1080, 1650), extra=''):
    z = rep['frames'][n]['zoom']; w = '+'.join(x['text'] for x in rep['frames'][n]['words'])
    im = Image.open(S / f'ours/o{n:03d}.png').convert('RGB').crop(box)
    k = K / z
    im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    return lab(im, f'OURS f{n:03d} {w} (x{k:.3f}, zoom {z:.3f}) {extra}')
rows = [
    (ref(142), ours(59)), (ref(78), ours(123)), (ref(146), ours(85)),
    (ref(113), ours(136)), (ref(55), ours(98)), (ref(32, (300, 300, 1650, 760)), ours(36, (0, 380, 1080, 900))),
    (ref(152), ours(142)), (ref(96, (200, 300, 1000, 800)), ours(97)),
]
W = max(a.width + b.width for a, b in rows) + 18; H = sum(max(a.height, b.height) for a, b in rows) + 6 * (len(rows) + 1)
G = Image.new('RGB', (W, H), (12, 12, 12)); y = 6
for a, b in rows:
    G.paste(a, (6, y)); G.paste(b, (12 + a.width, y)); y += max(a.height, b.height) + 6
G.save(LAB / 'analysis/critique_round1_samescale_typography.png')
# frame-height scale: ours whole frame at 1078 high next to ref full frames
pairs = [(142, 59), (113, 136), (32, 36), (55, 98)]
tiles = []
for r, o in pairs:
    a = lab(Image.open(LAB / f'ref/frames/f{r:03d}.png').convert('RGB'), f'REF f{r:03d}')
    b = Image.open(S / f'ours/o{o:03d}.png').convert('RGB').resize((606, 1078), Image.LANCZOS); lab(b, f'OURS f{o:03d}')
    t = Image.new('RGB', (a.width + b.width + 6, 1078), (12, 12, 12)); t.paste(a, (0, 0)); t.paste(b, (a.width + 6, 0)); tiles.append(t.resize((t.width // 2, t.height // 2), Image.LANCZOS))
G2 = Image.new('RGB', (tiles[0].width * 2 + 18, tiles[0].height * 2 + 18), (12, 12, 12))
for i, t in enumerate(tiles): G2.paste(t, (6 + (i % 2) * (t.width + 6), 6 + (i // 2) * (t.height + 6)))
G2.save(LAB / 'analysis/critique_round1_frameheight_scale.png')
