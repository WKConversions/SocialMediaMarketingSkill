"""Round-1 art-director critique evidence sheets (ours vs reference vs source)."""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
S = Path(sys.argv[1])          # scratch dir with ours/oNNN.png, src/sNNN.png
LAB = Path('/home/user/SocialMediaMarketingSkill/lab/caption-style-01')
OUT = LAB / 'analysis'
REF = LAB / 'ref/frames'
rep = json.load(open(LAB / 'out/user_v1.render.json'))
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf').exists() else ImageFont.load_default()

def label(im, txt, col=(255, 210, 0)):
    d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 30], fill=(20, 20, 20)); d.text((6, 4), txt, font=F, fill=col); return im

def ours(n): return Image.open(S / f'ours/o{n:03d}.png').convert('RGB')
def src(n): return Image.open(S / f'src/s{n:03d}.png').convert('RGB')
def ref(n): return Image.open(REF / f'f{n:03d}.png').convert('RGB')
def words(n): return ' + '.join(w['text'] for w in rep['frames'][n]['words'])

def grid(tiles, cols, pad=6, bg=(12, 12, 12)):
    w = max(t.width for t in tiles); h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    G = Image.new('RGB', (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, t in enumerate(tiles):
        G.paste(t, (pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)))
    return G

# 1) 12 frames spread across the clip, full frame, ours
frames = [9, 23, 36, 49, 59, 68, 77, 85, 96, 110, 123, 136]
tiles = [label(ours(n).resize((360, 640), Image.LANCZOS), f'f{n:03d} {words(n)} z{rep["frames"][n]["zoom"]:.3f}') for n in frames]
grid(tiles, 6).save(OUT / 'critique_round1_spread.png')

# 2) every frame of the Editen blur-in (f086-f101) and the opening (f000-f011) + Testvideo entrance (f026-f037), crop around text
def crop_text(im, box):
    return im.crop(box)
def entr(rng, box, scale, name):
    tiles = []
    for n in rng:
        t = crop_text(ours(n), box); t = t.resize((int(t.width * scale), int(t.height * scale)), Image.LANCZOS)
        tiles.append(label(t, f'f{n:03d} {words(n)} z{rep["frames"][n]["zoom"]:.3f}'))
    grid(tiles, 4).save(OUT / name)
entr(range(86, 102), (120, 1180, 960, 1720), 0.6, 'critique_round1_entrance_editen.png')
entr(range(0, 12), (240, 1120, 840, 1720), 0.6, 'critique_round1_entrance_dit.png')
entr(range(26, 38), (0, 300, 1080, 900), 0.5, 'critique_round1_entrance_testvideo.png')
entr(range(43, 55), (240, 1120, 840, 1720), 0.6, 'critique_round1_entrance_omte.png')
