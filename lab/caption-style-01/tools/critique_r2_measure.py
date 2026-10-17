#!/usr/bin/env python3
"""Round-2 critique measurements on out/user_v2.mp4.

    python3 critique_r2_measure.py OURS_DIR SRC_DIR

1. Untouched footage: render vs zoomed source outside the caption box (mean |diff|, per-channel bias).
2. Blend model inside the glyphs: render vs |C - zoomed source| (median error).
3. Text scale follows the zoom: ink bbox of the caption (pixels far from the zoomed source) per frame.
4. Speech onsets vs caption first frames (speech-band energy, 10 ms hop).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parent.parent


def zoomed(img, s):
    W, H = img.size
    if abs(s - 1) < 1e-5:
        return img
    box = (W / 2 - W / 2 / s, H / 2 - H / 2 / s, W / 2 + W / 2 / s, H / 2 + H / 2 / s)
    return img.resize((W, H), Image.BICUBIC, box=box)


def main():
    od, sd = Path(sys.argv[1]), Path(sys.argv[2])
    rep = json.load(open(LAB / 'out' / 'user_v2.render.json'))
    style = json.load(open(LAB / 'styles' / 'prime.json'))
    res = {'frames': []}
    for fr in rep['frames']:
        f = fr['frame']
        o = np.asarray(Image.open(od / f'f{f:03d}.png').convert('RGB'), np.float32) / 255
        s_img = zoomed(Image.open(sd / f'f{fr["source_frame"]:03d}.png').convert('RGB'), fr['zoom'])
        s = np.asarray(s_img, np.float32) / 255
        z = fr['zoom']
        w = fr['words'][0]
        bx = w['box']
        x0 = int(540 + (bx[0] - 540) * z) - 40
        x1 = int(540 + (bx[2] - 540) * z) + 40
        y0 = int(960 + (bx[1] - 960) * z) - 60
        y1 = int(960 + (bx[3] - 960) * z) + 60
        outside = np.ones(o.shape[:2], bool)
        outside[max(y0, 0):y1, max(x0, 0):x1] = False
        d = o - s
        ext = dict(mad=float(np.abs(d[outside]).mean() * 255), bias=[float(d[outside][:, c].mean() * 255) for c in range(3)])
        # glyph pixels: difference model
        unit = next(u for u in rep['units'] if u['text'] == w['text'])
        fill = style['blends'][unit['blend']].get('fill', '#FFFFFF').lstrip('#')
        C = np.array([int(fill[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255
        roi = (slice(max(y0, 0), y1), slice(max(x0, 0), x1))
        oo, ss = o[roi], s[roi]
        model = np.abs(C - ss) if unit['blend'] != 'normal' else np.broadcast_to(C, ss.shape)
        e_model = np.abs(oo - model).max(-1)
        e_src = np.abs(oo - ss).max(-1)
        glyph = (e_src > 0.15) & (e_model < e_src)
        core = glyph & (e_model < 0.5)
        # ink bbox in output
        ys, xs = np.nonzero(glyph)
        ink = [int(xs.min()) + roi[1].start, int(ys.min()) + roi[0].start, int(xs.max()) + roi[1].start, int(ys.max()) + roi[0].start] if len(xs) else None
        ink_luma = float((oo[glyph] @ np.array([0.2126, 0.7152, 0.0722])).mean()) if glyph.any() else None
        res['frames'].append(dict(frame=f, zoom=z, text=w['text'], outside=ext,
                                  glyph_px=int(glyph.sum()), model_err_med=float(np.median(e_model[glyph]) * 255) if glyph.any() else None,
                                  ink=ink, ink_luma=ink_luma))
    # summary
    mads = [r['outside']['mad'] for r in res['frames']]
    biases = np.array([r['outside']['bias'] for r in res['frames']])
    errs = [r['model_err_med'] for r in res['frames'] if r['model_err_med'] is not None]
    print('outside caption: mean |diff| levels  min %.2f  median %.2f  max %.2f' % (min(mads), np.median(mads), max(mads)))
    print('outside caption: mean bias per channel', biases.mean(0).round(2), 'max abs', np.abs(biases).max(0).round(2))
    print('glyph model error median levels: median %.1f  max %.1f' % (np.median(errs), max(errs)))
    # text scale vs zoom per unit (ink width ratio vs zoom ratio)
    by = {}
    for r in res['frames']:
        by.setdefault(r['text'], []).append(r)
    print('\nunit: ink-width ratio last/first vs zoom ratio last/first')
    for t, rs in by.items():
        a, b = rs[0], rs[-1]
        if a['ink'] and b['ink']:
            wa = a['ink'][2] - a['ink'][0]
            wb = b['ink'][2] - b['ink'][0]
            print(f'  {t:10s} frames {a["frame"]}-{b["frame"]}  ink w {wa}->{wb} ratio {wb / wa:.3f}  zoom ratio {b["zoom"] / a["zoom"]:.3f}  '
                  f'ink y {a["ink"][1]}-{a["ink"][3]} -> {b["ink"][1]}-{b["ink"][3]} (H {a["ink"][1] / 1920:.3f}-{b["ink"][3] / 1920:.3f})')
    json.dump(res, open(LAB / 'analysis' / 'critique_round2_measure.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
