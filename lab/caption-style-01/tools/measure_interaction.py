#!/usr/bin/env python3
"""Per-caption interaction metrics for a render (compare rounds against each other / the critique).

    python3 measure_interaction.py --report out/user_v2.render.json --source footage/user/clip.mp4 --style styles/prime.json

For every caption: background luma behind the letters (p5..p95, of the footage as zoomed in the output),
mean letter luma, mean |frame-to-frame change| inside the letters on frames whose zoom changed < 0.2%
(so the change comes from the footage moving, not from the zoom), the ink box after the zoom as a
fraction of the frame height, and the median model error |out - blend(B)| in 8-bit levels.
"""
import argparse
import json
import subprocess

import numpy as np
from PIL import Image


def read_all(video, W, H):
    cmd = ['ffmpeg', '-v', 'error', '-i', video, '-vf', 'scale=in_color_matrix=bt709:in_range=tv',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def zoomed(img, s, W, H):
    if abs(s - 1) < 1e-5:
        return img
    box = (W / 2 - W / 2 / s, H / 2 - H / 2 / s, W / 2 + W / 2 / s, H / 2 + H / 2 / s)
    return np.asarray(Image.fromarray(img).resize((W, H), Image.BICUBIC, box=box))


def luma(a):
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--style', required=True)
    ap.add_argument('--json', help='write the metrics here')
    a = ap.parse_args()
    rep = json.load(open(a.report))
    style = json.load(open(a.style))
    W, H = rep['width'], rep['height']
    src = read_all(a.source, W, H)
    out = read_all(rep['out'], W, H)
    res = {}
    prev = None
    for fr in rep['frames']:
        if not fr['words']:
            prev = None
            continue
        n = fr['frame'] - rep['first_frame']
        z = fr['zoom']
        o = out[n].astype(np.float32) / 255
        s = zoomed(src[fr['source_frame']], z, W, H).astype(np.float32) / 255
        w = fr['words'][0]
        u = next(x for x in rep['units'] if x['text'] == w['text'])
        b = style['blends'][u['blend']]
        C = np.array([int(b['fill'].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255
        model = np.abs(C - s) if b['mode'] == 'difference' else np.broadcast_to(C, s.shape)
        bx = w['box']
        cx, cy = W / 2, H / 2
        zb = [cx + (bx[0] - cx) * z, cy + (bx[1] - cy) * z, cx + (bx[2] - cx) * z, cy + (bx[3] - cy) * z]
        y0, y1, x0, x1 = max(int(zb[1]) - 4, 0), min(int(zb[3]) + 4, H), max(int(zb[0]) - 4, 0), min(int(zb[2]) + 4, W)
        oo, ss, mm = o[y0:y1, x0:x1], s[y0:y1, x0:x1], model[y0:y1, x0:x1]
        err = np.abs(oo - mm).mean(2)
        m = (err < 0.05) & (np.abs(oo - ss).mean(2) > 0.15)        # solid letter pixels
        if m.sum() < 300:
            prev = None
            continue
        rec = res.setdefault(w['text'], dict(frames=0, bg_p5=[], bg_p95=[], L=[], dL=[], err=[], box_frac=None,
                                             blend=u['blend'], cls=u['cls']))
        rec['frames'] += 1
        bl = luma(ss)[m]
        rec['bg_p5'].append(np.percentile(bl, 5))
        rec['bg_p95'].append(np.percentile(bl, 95))
        rec['L'].append(luma(oo)[m].mean())
        rec['err'].append(np.median(np.abs(oo - mm)[m]) * 255)
        top, bot = zb[1] / H, zb[3] / H
        rec['box_frac'] = [min(top, rec['box_frac'][0]) if rec['box_frac'] else top,
                           max(bot, rec['box_frac'][1]) if rec['box_frac'] else bot]
        if prev is not None and prev['text'] == w['text'] and abs(prev['zoom'] - z) < 0.002:
            mm2 = m & prev['m'] if prev['m'].shape == m.shape and prev['xy'] == (y0, x0) else None
            if mm2 is not None and mm2.sum() > 200:
                rec['dL'].append(np.abs(oo - prev['o']).mean(2)[mm2].mean() * 255)
        prev = dict(text=w['text'], zoom=z, m=m, o=oo, xy=(y0, x0))
    summary = {}
    for k, r in res.items():
        summary[k] = dict(cls=r['cls'], blend=r['blend'], frames=r['frames'],
                          bg_luma_p5=round(float(np.mean(r['bg_p5'])), 3), bg_luma_p95=round(float(np.mean(r['bg_p95'])), 3),
                          letter_luma=round(float(np.mean(r['L'])), 3),
                          dL_levels=round(float(np.mean(r['dL'])), 2) if r['dL'] else None, dL_frames=len(r['dL']),
                          model_err_levels=round(float(np.median(r['err'])), 2),
                          ink_box_after_zoom_H=[round(v, 3) for v in r['box_frac']])
        s = summary[k]
        print(f"{k:11s} {r['cls']:6s} bg luma p5..p95 {s['bg_luma_p5']:.3f}..{s['bg_luma_p95']:.3f}  letter luma {s['letter_luma']:.2f}  "
              f"frame-to-frame change {s['dL_levels'] if s['dL_levels'] is not None else float('nan'):5.2f} levels ({s['dL_frames']} pairs)  "
              f"model err {s['model_err_levels']:.1f}  ink box {s['ink_box_after_zoom_H'][0]:.3f}-{s['ink_box_after_zoom_H'][1]:.3f} H")
    allL = [v['letter_luma'] for v in summary.values()]
    print(f'mean letter luma over captions: {np.mean(allL):.3f}')
    if a.json:
        json.dump(summary, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
