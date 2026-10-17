#!/usr/bin/env python3
"""A/B review sheet for the opt-in variants: the glitch_rgb opening (first frames of the opening caption,
A = default glitch vs B = glitch_rgb) and the behind-subject depth word (B, 2x crops of the occlusion edge).

    python3 ab_sheet.py --a out/user_v2.render.json --b out/user_v2_ab.render.json --out out/ab_variants.png
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_sheets import font, hcat, labelled, read_frames, vcat, zoom_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    A, B = json.load(open(args.a)), json.load(open(args.b))
    W, H = A['width'], A['height']
    first = B['units'][0]
    gl = list(range(first['first'], first['first'] + 4))
    behind = next((u for u in B['units'] if u['behind_subject']), None)
    bf = [behind['first'] + k for k in (0, 4, 8, 14)] if behind else []
    fa = read_frames(A['out'], gl, W, H)
    fb = read_frames(B['out'], gl + bf, W, H)
    rows = []
    for name, fr in (('A (default glitch = footage on 2s)', fa), ('B (glitch_rgb, opt-in)', fb)):
        tiles = []
        for f in gl:
            r = next(x for x in (A if fr is fa else B)['frames'] if x['frame'] == f)
            box = zoom_box(r['words'][0]['box'], r['zoom'], W, H)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            crop = Image.fromarray(fr[f]).crop((int(cx - 220), int(cy - 110), int(cx + 220), int(cy + 110)))
            tiles.append(labelled(crop.resize((660, 330), Image.NEAREST), f'{name}  f{f:03d}', size=18))
        rows.append(hcat(tiles, gap=6))
    if behind:
        tiles = []
        for f in bf:
            r = next(x for x in B['frames'] if x['frame'] == f)
            w = next(x for x in r['words'] if x['text'] == behind['text'])
            box = zoom_box(w['box'], r['zoom'], W, H)
            crop = Image.fromarray(fb[f]).crop((int(box[0]) + 200, int(box[1]) + 60, int(box[0]) + 530, int(box[3]) + 60))
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.NEAREST)
            tiles.append(labelled(crop, f'B depth: "{behind["text"]}" behind the hair, f{f:03d} (2x)', size=18))
        rows.append(hcat(tiles, gap=6))
        full = [labelled(Image.fromarray(fb[f]).resize((W // 3, H // 3), Image.LANCZOS), f'B f{f:03d}', size=18) for f in bf]
        rows.append(hcat(full, gap=6))
    title = Image.new('RGB', (max(r.width for r in rows), 50), (22, 22, 22))
    ImageDraw.Draw(title).text((10, 10), 'Opt-in variants for the client to choose (not in the reference): RGB-split opening glitch, '
                               'depth word behind the head', fill=(255, 255, 255), font=font(24))
    vcat([title] + rows, gap=8).save(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
