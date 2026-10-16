"""Final summary sheet + metrics for the chosen fonts (Inter for the sans, Pinyon Script for the script)."""
import json
from fonts_fit import *
from fonts_script_round3 import WORDS as SWORDS
from fonts_script_hybrid import render_hybrid

A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
F = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/fonts/'
INTER = F + 'inter/Inter-Variable.ttf'
PINYON = F + 'pinyonscript/PinyonScript-Regular.ttf'
IMPERIAL = F + 'imperialscript/ImperialScript-Regular.ttf'
SANS = dict(axes={'wght': 750, 'opsz': 32}, features=['ss07'], size=113.5, track_em=-0.075)


def metrics():
    f = load(INTER, SANS['size'], SANS['axes'])
    s = dict(x_height=ink(f, 'x')[1], cap_H=ink(f, 'H')[1], asc_l=ink(f, 'l')[1], desc_g=ink(f, 'g')[3])
    p = load(PINYON, 216.4)
    q = dict(x_height=min(ink(p, 'u')[1], ink(p, 'n')[1]), asc_h=ink(p, 'h')[1], desc_g=ink(p, 'g')[3],
             **{f'cap_{c}': ink(p, c)[1] for c in 'PWRT'})
    return {k: round(float(v), 1) for k, v in s.items()}, {k: round(float(v), 1) for k, v in q.items()}


def main():
    sm, pm = metrics()
    print('Inter @113.5px', sm)
    print('Pinyon @216.4px', pm)
    tiles = []
    words = [('winning', 142, (750, 488, 1160, 612)), ('and', 146, (850, 488, 1065, 612)),
             ('charging', 78, (735, 495, 1180, 612)), ('still', 69, (860, 495, 1080, 612))]
    for word, n, box in words:
        m, cx = ref_geom(n, box)
        rm = render_mask(INTER, SANS['axes'], SANS['size'], SANS['track_em'], word, box, 579.5, cx, SANS['features'])
        s, dx, dy = iou(m, rm, 3); rm = np.roll(np.roll(rm, dy, 0), dx, 1)
        pair = stack([label(ref_crop(n, box), f'REF "{word}" f{n:03d}', size=16, h=22),
                      label(Image.fromarray((rm * 255).astype(np.uint8)), f'Inter wght750 opsz32 ss07 113.5px trk-0.075em IoU {s:.2f}', size=14, h=22)], gap=2)
        tiles.append(pair)
    row1 = Image.new('RGB', (sum(t.width for t in tiles) + 6 * 3, max(t.height for t in tiles)), (60, 60, 60)); x = 0
    for t in tiles:
        row1.paste(t, (x, 0)); x += t.width + 6
    stiles = []
    for word, n, box, base, xh, cx in SWORDS:
        a = render_hybrid(word, box, base, xh, cx, None, None, 1.0)
        b = render_hybrid(word, box, base, xh, cx, IMPERIAL, None, None) if word[0].isupper() else None
        col = [label(ref_crop(n, box), f'REF "{word}" f{n:03d}', size=16, h=22),
               label(a, f'Pinyon Script {xh * 216.4 / 83:.0f}px', size=16, h=22)]
        if b is not None:
            col.append(label(b, 'Pinyon + Imperial cap @2.5x x-height', size=16, h=22))
        stiles.append(stack(col, gap=2))
    r2 = stiles[:3]; r3 = stiles[3:]
    def hcat(cs):
        sh = Image.new('RGB', (sum(c.width for c in cs) + 6 * (len(cs) - 1), max(c.height for c in cs)), (60, 60, 60)); x = 0
        for c in cs:
            sh.paste(c, (x, 0)); x += c.width + 6
        return sh
    sheet = stack([row1, hcat(r2), hcat(r3)], gap=10, bg=(60, 60, 60))
    sheet.save(A + 'fonts_final_choice.png'); print(sheet.size)
    json.dump(dict(sans=dict(file=INTER, **SANS, metrics_px=sm), script=dict(file=PINYON, size_per_xheight=216.4 / 83, metrics_px_at_216=pm)),
              open(A + 'fonts_choice.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
