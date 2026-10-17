#!/usr/bin/env python3
"""Round-2 art-direction critique sheets: our render vs the reference, at the same pixel scale.

    python3 critique_r2_sheets.py OURS_FRAMES_DIR SRC_FRAMES_DIR OUT_PREFIX

OURS_FRAMES_DIR holds f000.png.. (0-based) from out/user_v2.mp4, SRC_FRAMES_DIR the same for the clip.
Reference frames come from ref/frames/fNNN.png (1-based). Both frames have a ~1080 px short side,
so crops are shown 1:1 (same pixel scale).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parent.parent
REF = LAB / 'ref' / 'frames'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except OSError:
        return ImageFont.load_default()


def ours(d, n):
    return Image.open(Path(d) / f'f{n:03d}.png').convert('RGB')


def ref(n):
    return Image.open(REF / f'f{n:03d}.png').convert('RGB')


def crop_c(img, cx, cy, w, h):
    return img.crop((int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)))


def label(img, text, sz=20, col=(255, 210, 0)):
    out = Image.new('RGB', (img.width, img.height + sz + 10), (18, 18, 18))
    out.paste(img, (0, sz + 10))
    ImageDraw.Draw(out).text((4, 3), text, font=font(sz), fill=col)
    return out


def hstack(ims, gap=6, bg=(30, 30, 30)):
    h = max(i.height for i in ims)
    w = sum(i.width for i in ims) + gap * (len(ims) - 1)
    out = Image.new('RGB', (w, h), bg)
    x = 0
    for i in ims:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def vstack(ims, gap=6, bg=(30, 30, 30)):
    w = max(i.width for i in ims)
    h = sum(i.height for i in ims) + gap * (len(ims) - 1)
    out = Image.new('RGB', (w, h), bg)
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def title(img, text, sz=26):
    return vstack([label(Image.new('RGB', (img.width, 4), (18, 18, 18)), text, sz, (255, 255, 255)), img], gap=0)


def main():
    od, sd, pre = sys.argv[1], sys.argv[2], sys.argv[3]
    rep = json.load(open(LAB / 'out' / 'user_v2.render.json'))
    spec = json.load(open(LAB / 'analysis' / 'reference_spec.json'))
    rwords = {w['id']: w for w in spec['words']}
    fr = {f['frame']: f for f in rep['frames']}

    # ---------------------------------------------------------------- spread: ours vs ref, same pixel scale
    pairs = [  # our frame, ref frame, note
        (2, 2, "dit / i'm"), (10, 6, 'dit / not'), (22, 25, 'is een (pink) / at my (pink)'),
        (31, 32, 'Testvideo / Prime'), (40, 35, 'Testvideo / Prime'), (50, 46, 'om te / at a'),
        (59, 78, 'kijken / charging'), (68, 128, 'of jij / in the'), (77, 69, 'goed / still'),
        (85, 146, 'kan / and'), (98, 56, 'Editen / hundred'), (108, 96, 'Editen / They'),
        (115, 123, 'wat ik / know'), (121, 142, 'graag / winning'), (126, 108, 'zou / me'),
        (131, 113, 'Willen / Win'), (142, 155, 'Willen / Regardless')]
    rows = []
    CW, CH = 900, 380
    for of, rf, note in pairs:
        o = ours(od, of)
        r = ref(rf)
        # centre crops on the caption: ours = report box mapped through the zoom; ref = frame centre
        w = fr[of]['words'][0]
        z = fr[of]['zoom']
        bx = w['box']
        cx = 540 + ((bx[0] + bx[2]) / 2 - 540) * z
        cy = 960 + ((bx[1] + bx[3]) / 2 - 960) * z
        oc = crop_c(o, cx, cy, CW, CH)
        rc = crop_c(r, 958, 560, CW, CH)
        th_o = o.resize((CH * 9 // 16, CH))
        th_r = r.resize((CH * 16 // 9 // 2, CH // 2))
        rows.append(label(hstack([th_o, oc, rc, th_r]),
                          f'OURS f{of:03d} "{w["text"]}" zoom {z:.3f}  |  REF f{rf:03d}   ({note})  crops 1:1, {CW}x{CH}px'))
    for i, chunk in enumerate([rows[:9], rows[9:]]):
        sheet = title(vstack(chunk), 'Round 2: ours (left, 9:16, 1080 px wide) vs reference (right, 16:9, 1078 px tall) at the same pixel scale')
        sheet.save(f'{pre}spread_{"ab"[i]}.png')

    # ---------------------------------------------------------------- entrances, every frame
    def strip(frames_o, frames_r, name_o, name_r, cy_o=None, w=560, h=330, ref_cy=560, ref_cx=958):
        ro = []
        for f in frames_o:
            o = ours(od, f)
            z = fr[f]['zoom']
            if fr[f]['words']:
                bx = fr[f]['words'][0]['box']
                cx = 540 + ((bx[0] + bx[2]) / 2 - 540) * z
                cy = 960 + ((bx[1] + bx[3]) / 2 - 960) * z
            else:
                cx, cy = 540, 960
            c = crop_c(o, cx, cy if cy_o is None else cy_o, w, h)
            txt = fr[f]['words'][0]['text'] if fr[f]['words'] else '-'
            op = fr[f]['words'][0]['opacity'] if fr[f]['words'] else 0
            ro.append(label(c, f'ours f{f:03d} {txt} a={op:.2f} z={z:.3f} src{fr[f]["source_frame"]}', 17))
        rr = [label(crop_c(ref(f), ref_cx, ref_cy, w, h), f'ref f{f:03d}', 17) for f in frames_r]
        return ro, rr

    # Editen blur-in vs They blur-in
    ro, rr = strip(list(range(86, 100)), list(range(84, 98)), 'Editen', 'They', ref_cx=540, ref_cy=520)
    g = vstack([hstack(ro[i:i + 7]) for i in range(0, 14, 7)] + [hstack(rr[i:i + 7]) for i in range(0, 14, 7)])
    title(g, 'Entrance: ours "kan" -> "Editen" blur_in (f086-f099, top two rows) vs ref "They" blur-in (f084-f097, bottom two rows)').save(f'{pre}entrance_editen.png')

    # dit opening + wat ik "glitch" vs ref opening
    ro, rr = strip(list(range(0, 9)), list(range(1, 10)), 'dit', "i'm")
    ro2, _ = strip(list(range(109, 118)), [], 'wat ik', '')
    g = vstack([hstack(ro[:5]), hstack(ro[5:9]), hstack(ro2[:5]), hstack(ro2[5:9]), hstack(rr[:5]), hstack(rr[5:9])])
    title(g, 'Entrances: ours "dit" f000-f008 and "wat ik" f109-f117 ("glitch" = footage on 2s) vs ref opening f001-f009').save(f'{pre}entrance_dit_watik.png')

    # Testvideo cut-in + punch vs Prime
    ro, rr = strip(list(range(26, 36)), list(range(26, 36)), 'Testvideo', 'Prime', w=1000, h=380)
    g = vstack([hstack(ro[:5]), hstack(ro[5:10]), hstack(rr[:5]), hstack(rr[5:10])])
    title(g, 'Entrance + punch-in: ours "is een" -> "Testvideo" f026-f035 vs ref "at my" -> "Prime" f026-f035').save(f'{pre}entrance_testvideo.png')


if __name__ == '__main__':
    main()
