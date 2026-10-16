"""Script round-1b/2: all reference script words (incl. swash capitals) vs candidates,
everything normalised to the same x-height."""
import sys, json
from fonts_lib import *
from fonts_script_fit import ref_x, fam_path

A = '/home/user/SocialMediaMarketingSkill/lab/caption-style-01/analysis/'
XN = 46  # normalised x-height in px
REFS = [  # word, frame, box, measured x-height px
    ('Prime', 32, (560, 370, 1420, 720), 110),
    ('hundred', 55, (600, 455, 1380, 650), 83),
    ('They', 96, (250, 330, 800, 640), 75),
    ('Win', 113, (760, 380, 1320, 680), 90),
    ('Regardless', 152, (620, 440, 1340, 680), 54),
]


def ref_tiles():
    out = []
    for w, n, box, xh in REFS:
        im = ref_crop(n, box); s = XN / xh
        out.append(im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS))
    return out


def cand_tile(path, word, W, H, axes=None, stroke=0, features=None):
    xh, _ = ref_x(path)
    size = XN / xh
    f = load(path, size * 2, axes)
    im = Image.new('L', (W * 2 * 2, H * 2), 0)
    d = ImageDraw.Draw(im)
    d.text((W * 2, H * 2 * 0.62), word, font=f, fill=255, anchor='ms', features=features,
           stroke_width=int(stroke * 2), stroke_fill=255)
    a = np.asarray(im); cols = np.nonzero(a.max(0) > 30)[0]
    if len(cols):
        c = (cols.min() + cols.max()) // 2
        im = im.crop((c - W, 0, c + W, H * 2))
    return im.resize((W, H), Image.LANCZOS)


def sheet(specs, out, extra=None):
    refs = ref_tiles()
    H = max(r.height for r in refs)
    widths = [max(r.width, 10) + 40 for r in refs]
    widths[0] += 60  # Prime swash
    rows = []
    row = [label(Image.new('RGB', (widths[i], H)), '') for i in range(len(refs))]
    row = []
    for i, r in enumerate(refs):
        t = Image.new('RGB', (widths[i], H), (0, 0, 0)); t.paste(r, ((widths[i] - r.width) // 2, 0))
        row.append(label(t, f'REF {REFS[i][0]} f{REFS[i][1]:03d}', size=16, h=22))
    rows.append(row)
    for spec in specs:
        if isinstance(spec, tuple):
            name, path, axes, stroke, feats = spec
        else:
            name, path, axes, stroke, feats = spec, fam_path(spec), ({'opsz': 72} if spec == 'ballet' else None), 0, None
        row = []
        for i, (w, n, box, xh) in enumerate(REFS):
            t = cand_tile(path, w, widths[i], H, axes, stroke, feats)
            row.append(label(t, name if i == 0 else '', size=16, h=22))
        rows.append(row)
    lines = []
    for row in rows:
        Wt = sum(t.width for t in row) + 4 * (len(row) - 1)
        L = Image.new('RGB', (Wt, row[0].height), (60, 60, 60)); x = 0
        for t in row:
            L.paste(t, (x, 0)); x += t.width + 4
        lines.append(L)
    sh = stack(lines, bg=(60, 60, 60), gap=4)
    sh.save(out); print(out, sh.size)


if __name__ == '__main__':
    pass
