"""Swash-capital comparison: reference P/W/R/T crops vs candidate capitals, all at equal height."""
from fonts_lib import *
from fonts_script_fit import fam_path

A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
H = 260
CAPS = [('P', 32, (575, 380, 1000, 705)), ('W', 113, (775, 395, 1130, 670)), ('R', 152, (628, 448, 905, 645)),
        ('T', 96, (255, 335, 700, 600))]


def ref_tiles():
    out = []
    for c, n, box in CAPS:
        im = ref_crop(n, box)
        s = H / im.height
        out.append(im.resize((int(im.width * s), H), Image.LANCZOS))
    return out


def cap_tile(path, ch, W, features=None):
    f = load(path, 400)
    x0, top, x1, bottom = ink(f, ch, features)
    hh = top + bottom
    size = 400 * (H - 16) / hh
    f = load(path, size)
    im = Image.new('L', (W * 3, H), 0); d = ImageDraw.Draw(im)
    x0, top, x1, bottom = ink(f, ch, features)
    d.text((W * 1.5 - (x0 + x1) / 2, 8 + top), ch, font=f, fill=255, anchor='ls', features=features)
    return im.crop((W, 0, 2 * W, H))


def sheet(specs, out):
    refs = ref_tiles()
    widths = [r.width + 30 for r in refs]
    rows = []
    row = []
    for r, w, (c, n, _) in zip(refs, widths, CAPS):
        t = Image.new('RGB', (w, H)); t.paste(r, ((w - r.width) // 2, 0)); row.append(label(t, f'REF {c} f{n:03d}', size=16, h=22))
    rows.append(row)
    for name, spec, feats in specs:
        p = fam_path(spec)
        row = [label(cap_tile(p, c, w, feats), name if i == 0 else '', size=16, h=22) for i, ((c, n, b), w) in enumerate(zip(CAPS, widths))]
        rows.append(row)
    lines = []
    for row in rows:
        L = Image.new('RGB', (sum(t.width for t in row) + 4 * (len(row) - 1), row[0].height), (60, 60, 60)); x = 0
        for t in row:
            L.paste(t, (x, 0)); x += t.width + 4
        lines.append(L)
    # two columns of rows to keep the sheet viewable
    half = (len(lines) + 2) // 2
    c1 = stack(lines[:half], bg=(60, 60, 60), gap=4); c2 = stack([lines[0]] + lines[half:], bg=(60, 60, 60), gap=4)
    sh = Image.new('RGB', (c1.width + c2.width + 12, max(c1.height, c2.height)), (60, 60, 60))
    sh.paste(c1, (0, 0)); sh.paste(c2, (c1.width + 12, 0)); sh.save(out); print(out, sh.size)


if __name__ == '__main__':
    specs = [('pinyonscript', 'pinyonscript', None), ('greatvibes', 'greatvibes', None),
             ('greatvibes ss02', 'greatvibes', ['ss02']), ('greatvibes ss03', 'greatvibes', ['ss03']),
             ('greatvibes ss04', 'greatvibes', ['ss04']), ('monsieurladoulaise', 'monsieurladoulaise', None),
             ('herrvonmuellerhoff', 'herrvonmuellerhoff', None), ('mrdehaviland', 'mrdehaviland', None),
             ('imperialscript', 'imperialscript', None), ('alexbrush', 'alexbrush', None),
             ('italianno ss02', 'italianno', ['ss02']), ('allura ss02', 'allura', ['ss02']),
             ('beaurivage ss01', 'beaurivage', ['ss01']), ('birthstone ss02', 'birthstone', ['ss02']),
             ('missfajardose', 'missfajardose', None), ('ephesis', 'ephesis', None)]
    sheet(specs, A + 'fonts_script_round2_capitals.png')
