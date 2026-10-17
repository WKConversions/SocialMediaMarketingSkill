#!/usr/bin/env python3
"""Review sheets for a caption render: reference-vs-ours comparison and the 'interaction' strip.

    python3 make_sheets.py --report out/user_v1.render.json --source footage/user/clip.mp4 \
        --ref-frames ref/frames --comparison out/comparison.png --interaction out/interaction.png

Uses the per-frame report written by render_captions.py (which caption is on which frame, its box before
the digital zoom, the zoom factor) to pick comparable frames automatically.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def font(size, bold=True):
    p = '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else '')
    return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()


def read_frames(video, indices, W, H):
    """Decode the given 0-based frame indices (bt709 aware) -> {index: HxWx3 uint8}."""
    idx = sorted(set(indices))
    sel = '+'.join(f'eq(n\\,{i})' for i in idx)
    cmd = ['ffmpeg', '-v', 'error', '-i', video, '-vf', f"select='{sel}',scale=in_color_matrix=bt709:in_range=tv",
           '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    return {i: arr[k] for k, i in enumerate(idx)}


def zoom_box(box, s, W, H, origin=(0.5, 0.5)):
    ox, oy = origin[0] * W, origin[1] * H
    return [ox + (box[0] - ox) * s, oy + (box[1] - oy) * s, ox + (box[2] - ox) * s, oy + (box[3] - oy) * s]


def labelled(img, text, h=34, size=22, colour=(255, 210, 0)):
    out = Image.new('RGB', (img.width, img.height + h), (22, 22, 22))
    out.paste(img, (0, h))
    ImageDraw.Draw(out).text((8, 5), text, fill=colour, font=font(size))
    return out


def hcat(ims, gap=10, bg=(40, 40, 40)):
    out = Image.new('RGB', (sum(i.width for i in ims) + gap * (len(ims) - 1), max(i.height for i in ims)), bg)
    x = 0
    for i in ims:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def vcat(ims, gap=14, bg=(40, 40, 40)):
    out = Image.new('RGB', (max(i.width for i in ims), sum(i.height for i in ims) + gap * (len(ims) - 1)), bg)
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def crop_around(img, box, w, h):
    """Crop w x h (clamped to the frame) centred on box."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(min(max(cx - w / 2, 0), img.width - w))
    y0 = int(min(max(cy - h / 2, 0), img.height - h))
    return img.crop((x0, y0, x0 + w, y0 + h)), (x0, y0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', required=True)
    ap.add_argument('--source', required=True, help='the uncaptioned source video')
    ap.add_argument('--ref-frames', default=os.path.join(LAB, 'ref/frames'))
    ap.add_argument('--comparison', required=True)
    ap.add_argument('--interaction', required=True)
    args = ap.parse_args()

    rep = json.load(open(args.report))
    W, H, fps = rep['width'], rep['height'], rep['fps']
    frames = rep['frames']
    units = rep['units']
    by_text = {u['text']: u for u in units}

    def frames_of(text):
        return [r for r in frames if any(w['text'] == text for w in r['words'])]

    def word_rec(r, text):
        return next(w for w in r['words'] if w['text'] == text)

    # ---------- pick our frames
    src_needed = set(r['frame'] for r in frames)
    src = read_frames(args.source, src_needed, W, H)
    out_all = read_frames(rep['out'], [r['frame'] - rep['first_frame'] for r in frames], W, H)
    out_f = {r['frame']: out_all[r['frame'] - rep['first_frame']] for r in frames}

    def lum_under(r, text):
        b = [int(v) for v in word_rec(r, text)['box']]
        reg = src[r['frame']][max(b[1], 0):b[3], max(b[0], 0):b[2]].astype(np.float32)
        return reg.mean() / 255.0

    def motion_under(r, text):
        b = [int(v) for v in word_rec(r, text)['box']]
        f = r['frame']
        if f + 1 not in src:
            return 0.0
        a = src[f][max(b[1], 0):b[3], max(b[0], 0):b[2]].astype(np.float32)
        c = src[f + 1][max(b[1], 0):b[3], max(b[0], 0):b[2]].astype(np.float32)
        return float(np.abs(a - c).mean() / 255.0)

    sans = [u for u in units if u['cls'] == 'sans']
    settled = lambda text: [r for r in frames_of(text) if word_rec(r, text)['opacity'] >= 1.0 and word_rec(r, text)['k'] >= 1]
    dark_pick = min(((lum_under(r, u['text']), r['frame'], u['text']) for u in sans for r in settled(u['text'])))
    bright_pick = max(((lum_under(r, u['text']), r['frame'], u['text']) for u in sans for r in settled(u['text'])
                       if abs(r['zoom'] - 1) < 0.02))
    moving_pick = max(((motion_under(r, u['text']), r['frame'], u['text']) for u in units if not u['behind_subject']
                       for r in settled(u['text'])))
    script_units = [u for u in units if u['cls'] == 'script']
    pink = next((u for u in script_units if u['blend'] != 'difference'), script_units[0])
    pink_frame = frames_of(pink['text'])[len(frames_of(pink['text'])) // 2]['frame']
    glitch_u = next((u for u in units if u['entrance'] == 'glitch'), units[0])
    glitch_frame = glitch_u['first'] + 1
    fade_u = next((u for u in units if u['entrance'].startswith('blur_in')), None)
    fade_frame = fade_u['first'] + 2 if fade_u else None

    pairs = [
        (69, 'REF f069 "still": sans over a dark room -> near-white letters',
         dark_pick[1], f'OURS f{dark_pick[1]:03d} "{dark_pick[2]}": sans over the darkest background'),
        (123, 'REF f123 "know": sans over sky and jacket -> letters invert per region',
         bright_pick[1], f'OURS f{bright_pick[1]:03d} "{bright_pick[2]}": sans over the brightest background'),
        (32, 'REF f032 "Prime": swash script, pink #E28B8B difference',
         pink_frame, f'OURS f{pink_frame:03d} "{pink["text"]}": script, {"pink " if pink["blend"] != "difference" else ""}difference'
                     + (', behind the subject (depth test)' if pink['behind_subject'] else ', in front, inverting what is behind it')),
        (13, 'REF f013 "even": watch/hand moving behind the letters',
         moving_pick[1], f'OURS f{moving_pick[1]:03d} "{moving_pick[2]}": most motion behind a caption'),
        (3, 'REF f003 "i\'m": "glitch" = footage on 2s + inverted LED jump (no RGB split)',
         glitch_frame, f'OURS f{glitch_frame:03d} "{glitch_u["text"]}": glitch entrance (footage held in pairs)'),
    ]
    if fade_u:
        pairs.append((88, 'REF f088 "They": blur-in entrance, 3rd frame (opacity ~0.4)',
                      fade_frame, f'OURS f{fade_frame:03d} "{fade_u["text"]}": same blur-in curve, 3rd frame'))

    rows = []
    RH = 600
    for ref_n, ref_lab, our_f, our_lab in pairs:
        ref = Image.open(os.path.join(args.ref_frames, f'f{ref_n:03d}.png')).convert('RGB')
        ref_s = ref.resize((int(ref.width * RH / ref.height), RH), Image.LANCZOS)
        ours = Image.fromarray(out_f[our_f])
        ours_s = ours.resize((int(W * RH / H), RH), Image.LANCZOS)
        r = next(x for x in frames if x['frame'] == our_f)
        if r['words']:
            boxes = [zoom_box(w['box'], r['zoom'], W, H) for w in r['words']]
            box = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        else:
            box = [0, H * 0.7, W, H * 0.85]
        crop, _ = crop_around(ours, box, min(W, 1000), min(H, int(RH * 1.0)))
        crop = crop.resize((int(crop.width * RH / crop.height), RH), Image.LANCZOS)
        row = hcat([labelled(ref_s, ref_lab, size=20), labelled(ours_s, 'OURS (full frame)', size=20),
                    labelled(crop, our_lab, size=20)])
        rows.append(row)
    title = Image.new('RGB', (rows[0].width, 60), (22, 22, 22))
    from math import gcd
    g = gcd(W, H)
    aspect = f'{W // g}:{H // g}' if W // g < 50 else f'{W / H:.2f}:1'
    ImageDraw.Draw(title).text((10, 12), f'Prime caption style: reference (Toby de Boer reel, 16:9) vs ours on '
                               f'{os.path.basename(rep["video"])} ({W}x{H}, {aspect}, layout {rep.get("layout")})',
                               fill=(255, 255, 255), font=font(30))
    vcat([title] + rows).save(args.comparison)
    print('comparison:', args.comparison, [p[2] for p in pairs])

    # ---------- interaction strip: 6 consecutive frames of one caption while the footage moves behind it
    def letter_change(r0, r1, text):
        # mean |change| of our output inside the letters (pixels the caption altered) between two frames
        b = [int(v) for v in word_rec(r0, text)['box']]
        sl = (slice(max(b[1], 0), b[3]), slice(max(b[0], 0), b[2]))
        o0, o1 = out_f[r0['frame']][sl].astype(np.float32), out_f[r1['frame']][sl].astype(np.float32)
        m = (np.abs(o0 - src[r0['frame']][sl].astype(np.float32)).mean(2) > 60) & \
            (np.abs(o1 - src[r1['frame']][sl].astype(np.float32)).mean(2) > 60)
        return float(np.abs(o1 - o0).mean(2)[m].mean()) if m.sum() > 200 else 0.0

    best = None
    for u in units:
        if u['behind_subject']:
            continue
        rs = [r for r in settled(u['text']) if abs(r['zoom'] - 1) < 0.01]
        for k in range(len(rs) - 5):
            win = rs[k:k + 6]
            if win[-1]['frame'] - win[0]['frame'] != 5:
                continue
            m = sum(letter_change(a, b, u['text']) for a, b in zip(win[:-1], win[1:]))
            if best is None or m > best[0]:
                best = (m, u['text'], [r['frame'] for r in win])
    _, text, fids = best
    tiles_out, tiles_src, swatches = [], [], []
    r0 = next(x for x in frames if x['frame'] == fids[0])
    box = word_rec(r0, text)['box']
    pad = 30
    cb = (int(box[0]) - pad, int(box[1]) - pad, int(box[2]) + pad, int(box[3]) + pad)
    for f in fids:
        o = out_f[f].astype(np.float32)
        s = src[f].astype(np.float32)
        oc, sc = o[max(cb[1], 0):cb[3], max(cb[0], 0):cb[2]], s[max(cb[1], 0):cb[3], max(cb[0], 0):cb[2]]
        tiles_out.append(labelled(Image.fromarray(oc.astype(np.uint8)), f'f{f:03d}  t={f / fps:.3f}s', size=20))
        tiles_src.append(labelled(Image.fromarray(sc.astype(np.uint8)), f'footage behind, f{f:03d}', size=18, colour=(180, 180, 180)))
        # inside-letter colour: pixels where the caption changed the frame a lot
        diff = np.abs(oc - sc).mean(2)
        mask = diff > 60
        cols = []
        for part in np.array_split(np.arange(oc.shape[1]), 4):
            mm = mask[:, part]
            cols.append(oc[:, part][mm].mean(0) if mm.any() else np.zeros(3))
        sw = Image.new('RGB', (oc.shape[1], 44), (22, 22, 22))
        d = ImageDraw.Draw(sw)
        qw = oc.shape[1] / 4
        for q, c in enumerate(cols):
            d.rectangle([q * qw + 3, 4, (q + 1) * qw - 3, 40], fill=tuple(int(v) for v in c))
        swatches.append(sw)
    strip = vcat([hcat(tiles_out, gap=6), hcat(swatches, gap=6), hcat(tiles_src, gap=6)], gap=6)
    zs = [next(x for x in frames if x['frame'] == f)['zoom'] for f in fids]
    zlab = f'zoom {min(zs):.3f}-{max(zs):.3f}' if max(zs) - min(zs) >= 0.0005 else f'zoom {zs[0]:.3f}'
    lines = [(f'Interaction: "{text}" over 6 consecutive frames ({zlab}, so almost only the footage moves)', 26, True, (255, 255, 255)),
             ('row 1: our output | row 2: mean colour inside the letters, left->right quarters | row 3: the footage behind', 20, False, (200, 200, 200)),
             (f'mean change inside the letters: {best[0] / 5:.1f} levels per frame', 20, False, (200, 200, 200))]
    fitted = []
    for txt, size, bold, col in lines:          # shrink each line until it fits the strip width
        while size > 10 and font(size, bold).getlength(txt) > strip.width - 20:
            size -= 1
        fitted.append((txt, font(size, bold), col, size))
    title = Image.new('RGB', (strip.width, sum(f[3] + 12 for f in fitted) + 8), (22, 22, 22))
    d = ImageDraw.Draw(title)
    y = 8
    for txt, fnt, col, size in fitted:
        d.text((10, y), txt, fill=col, font=fnt)
        y += size + 12
    vcat([title, strip], gap=0).save(args.interaction)
    print('interaction:', args.interaction, text, fids)


if __name__ == '__main__':
    main()
