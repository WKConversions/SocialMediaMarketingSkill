"""Round-2 sheets for the sans: grayscale side-by-side and red/cyan overlays of the finalists."""
import json
from fonts_fit import *
from fonts_overlay import overlay
A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
fit = json.load(open(A + 'cache_sans_fit.json'))
fit['Inter (opsz32, wght750, ss07) CHOSEN'] = dict(path=font_path('inter'), axes={'wght': 750, 'opsz': 32}, features=['ss07'],
                                              stroke=0, size=113.5, track_em=-0.075)
order = ['Inter (opsz32, wght750, ss07) CHOSEN', 'intertight', 'instrumentsans', 'onest', 'monasans', 'plusjakartasans',
         'golostext', 'geist', 'FreeSansBold+stroke(system,GPL)']
WORDS = [('winning', 142, (750, 488, 1160, 612)), ('and', 146, (850, 488, 1065, 612)), ('still', 69, (860, 495, 1080, 612)),
         ('charging', 78, (735, 488, 1180, 612)), ('yet', 22, (860, 488, 1055, 612)), ('at a', 46, (820, 488, 1100, 612)),
         ('know', 124, (810, 488, 1110, 612))]


def cand_tile(nm, word, n, box, cx, zoom):
    v = fit[nm]
    rm = render_mask(v['path'], v['axes'], v['size'], v['track_em'], word, box, 579.5, cx, v.get('features'), v.get('stroke', 0))
    return rm


def main(zoom=2):
    cols = []
    for word, n, box in WORDS:
        m, cx = ref_geom(n, box)
        if word in ('at a', 'know', 'yet'):
            cx = 957.0  # bright background / scaling: use the frame-centre convention
        ref = ref_crop(n, box)
        tiles = [label(ref.resize((ref.width * zoom, ref.height * zoom), Image.LANCZOS), f'REF "{word}" f{n:03d}', size=20, h=28)]
        for nm in order:
            rm = cand_tile(nm, word, n, box, cx, zoom)
            if word not in ('at a', 'know'):
                s, dx, dy = iou(m, rm, 3); rm = np.roll(np.roll(rm, dy, 0), dx, 1); t = f'{nm[:30]}  IoU {s:.3f}'
            else:
                t = nm[:30]
            im = Image.fromarray((rm * 255).astype(np.uint8))
            tiles.append(label(im.resize((im.width * zoom, im.height * zoom), Image.LANCZOS), t, size=20, h=28))
        cols.append(stack(tiles, gap=3))
    # two sheets: first 4 words, last 3
    for name, cs in (('fonts_sans_round2a.png', cols[:4]), ('fonts_sans_round2b.png', cols[4:])):
        W = sum(c.width for c in cs) + 8 * (len(cs) - 1)
        sh = Image.new('RGB', (W, max(c.height for c in cs)), (60, 60, 60)); x = 0
        for c in cs:
            sh.paste(c, (x, 0)); x += c.width + 8
        sh.save(A + name); print(A + name, sh.size)
    # overlay sheet (red = reference only, cyan = candidate only, white = both)
    ocols = []
    for word, n, box in WORDS[:5]:
        m, cx = ref_geom(n, box)
        tiles = [label(overlay(m, m * 0, 2), f'REF mask "{word}"', size=20, h=28)]
        for nm in order[:3] + ['FreeSansBold+stroke(system,GPL)']:
            rm = cand_tile(nm, word, n, box, cx, 2)
            s, dx, dy = iou(m, rm, 3); rm = np.roll(np.roll(rm, dy, 0), dx, 1)
            tiles.append(label(overlay(m, rm, 2), f'{nm[:30]} IoU {s:.3f}', size=20, h=28))
        ocols.append(stack(tiles, gap=3))
    W = sum(c.width for c in ocols) + 8 * (len(ocols) - 1)
    sh = Image.new('RGB', (W, max(c.height for c in ocols)), (60, 60, 60)); x = 0
    for c in ocols:
        sh.paste(c, (x, 0)); x += c.width + 8
    sh.save(A + 'fonts_sans_overlay.png'); print(A + 'fonts_sans_overlay.png', sh.size)


if __name__ == '__main__':
    main()
