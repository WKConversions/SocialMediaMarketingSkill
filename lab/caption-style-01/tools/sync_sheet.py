#!/usr/bin/env python3
"""Speech-band energy envelope vs caption on-screen spans (one or more renders) vs whisper word spans.

    python3 sync_sheet.py --video footage/user/clip.mp4 --words footage/user/clip_words.json \
        --report out/user_v1.render.json --report out/user_v2.render.json --style styles/prime.json --out sync.png
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_plan import speech_energy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--words', required=True)
    ap.add_argument('--report', action='append', required=True)
    ap.add_argument('--style', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.style)).get('plan_rules', {}).get('sync', {})
    db, floor, hop = speech_energy(a.video, cfg)
    thr = cfg.get('voiced_db_above_floor', 15)
    reps = [json.load(open(r)) for r in a.report]
    words = json.load(open(a.words))
    dur = len(db) * hop
    W = 2400
    px = W / dur
    H = 300 + 90 * len(reps) + 90
    im = Image.new('RGB', (W, H), (16, 16, 16))
    d = ImageDraw.Draw(im)
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    for t in np.arange(0, dur, 0.1):
        X = int(t * px)
        d.line([X, 0, X, H], fill=(70, 70, 70) if round(t * 10) % 5 == 0 else (36, 36, 36))
        if round(t * 10) % 5 == 0:
            d.text((X + 2, H - 22), f'{t:.1f}s', font=F, fill=(150, 150, 150))
    e = np.clip(db - floor, 0, 35) / 35
    for i in range(len(db)):
        X = int(i * hop * px)
        d.line([X, 240, X, 240 - int(e[i] * 200)], fill=(80, 170, 255) if db[i] - floor > thr else (50, 90, 140))
    yt = 240 - int(thr / 35 * 200)
    d.line([0, yt, W, yt], fill=(255, 80, 80))
    d.text((6, 6), f'speech-band energy {cfg.get("band_hz")} Hz, dB above floor (bright = voiced, red line = floor + {thr} dB)',
           font=F, fill=(80, 170, 255))
    cols = [(255, 210, 0), (255, 140, 60)]
    y = 260
    for r, path in zip(reps, a.report):
        fps = r['fps']
        d.text((6, y), f'caption on screen: {os.path.basename(path)}', font=F, fill=(255, 210, 0))
        for k, u in enumerate(r['units']):
            x0, x1 = u['first'] / fps * px, (u['last'] + 1) / fps * px
            d.rectangle([int(x0), y + 22, int(x1) - 2, y + 72], fill=cols[k % 2])
            d.text((int(x0) + 4, y + 36), u['text'], font=F, fill=(0, 0, 0))
        y += 90
    d.text((6, y), 'whisper word spans', font=F, fill=(120, 220, 120))
    for k, w in enumerate(words):
        d.rectangle([int(w['start'] * px), y + 22, int(w['end'] * px) - 2, y + 66], fill=(120, 220, 120) if k % 2 else (60, 160, 60))
        d.text((int(w['start'] * px) + 3, y + 34), w['word'], font=F, fill=(0, 0, 0))
    im.save(a.out)
    print(a.out)


if __name__ == '__main__':
    main()
