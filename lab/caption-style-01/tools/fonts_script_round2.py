"""Round-2 script sheet: 'hundred' f055 reference vs fitted finalists (grayscale + red/cyan overlay)."""
import json
from fonts_script_fit2 import *
from fonts_overlay import overlay
fit = json.load(open(A + 'cache_script_fit2.json'))
names = [k for k, v in sorted(fit.items(), key=lambda kv: -kv[1]['iou'])][:6]
m, cx = refs_hundred()
ref = ref_crop(55, BOX)
tiles = [label(ref, 'REFERENCE "hundred" f055 (1:1 pixels)', size=20, h=28)]
otiles = [label(overlay(m, m * 0, 1), 'REF mask', size=20, h=28)]
for nm in names:
    v = fit[nm]
    rm = render_script(v['path'], v['axes'], v['size'], v['track_em'], 'hundred', BOX, BASE, cx, v['shear_deg'], v['stroke'])
    s, dx, dy = iou(m, rm, 4); rm = np.roll(np.roll(rm, dy, 0), dx, 1)
    t = f"{nm} {v['size']:.0f}px shear+{v['shear_deg']:.1f}deg stroke{v['stroke']:.0f}px IoU {s:.3f}"
    tiles.append(label(Image.fromarray((rm * 255).astype(np.uint8)), t, size=20, h=28))
    otiles.append(label(overlay(m, rm, 1), t, size=20, h=28))
a = stack(tiles, gap=4); b = stack(otiles, gap=4)
sh = Image.new('RGB', (a.width + b.width + 8, max(a.height, b.height)), (60, 60, 60)); sh.paste(a, (0, 0)); sh.paste(b, (a.width + 8, 0))
sh.save(A + 'fonts_script_round2_hundred.png'); print(sh.size)
