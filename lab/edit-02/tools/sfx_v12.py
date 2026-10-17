"""Sound for version 12: one sound per moment, following the client's sound reference.

shutter  every caption (client's own file), biggest wave on the caption's first frame
hit      under script words, starting on the caption frame
flash    quick cuts: a bright, short burst whose peak sits on the cut frame
whoosh   scene to scene (into/out of the canvas): swells and peaks on the cut
glitch   a big text graphic appears: short digital burst starting on its first frame
click    an element pops up on the canvas: tiny UI click starting on its first frame
riser    builds over ~1 s and peaks on the moment it leads into
All synthesised sounds are mechanical/physical; nothing glowy (client rule).

Usage: python3 sfx_v12.py EVENTS.json DURATION_S OUT.wav
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
from sfx_prime import (SR, bandpass_noise, db, hit, place, place_onset, place_peak, reverb, rng,  # noqa: E402
                       shutter_peak, whoosh, write_wav)


def flash(dur=0.16):
    n = int(dur * SR)
    t = np.arange(n) / SR
    burst = bandpass_noise(n, 1500, 14000) * np.exp(-t / 0.035) * (1 - np.exp(-t / 0.002))
    thump = np.sin(2 * np.pi * (70 + 60 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.05)
    x = burst + 0.5 * thump
    return x / np.abs(x).max()


def click():
    n = int(0.025 * SR)
    t = np.arange(n) / SR
    x = bandpass_noise(n, 3000, 9000) * np.exp(-t / 0.0015) + 0.6 * np.sin(2 * np.pi * 2400 * t) * np.exp(-t / 0.003)
    return x / np.abs(x).max()


def glitch(dur=0.22):
    out, i = [], 0
    n_total = int(dur * SR)
    while i < n_total:
        seg = int(rng.uniform(0.008, 0.026) * SR)
        t = np.arange(seg) / SR
        kind = rng.integers(0, 4)
        if kind == 0:                                     # square-wave blip
            s = np.sign(np.sin(2 * np.pi * rng.uniform(300, 2500) * t))
        elif kind == 1:                                   # bit-crushed noise
            s = np.round(rng.standard_normal(seg) * 2) / 3
            s = np.repeat(s[::12], 12)[:seg]
        elif kind == 2:                                   # sample-and-hold buzz
            s = np.repeat(rng.uniform(-1, 1, seg // 40 + 1), 40)[:seg]
        else:                                             # dropout
            s = np.zeros(seg)
        out.append(s * rng.uniform(0.5, 1.0))
        i += seg
    x = np.concatenate(out)[:n_total]
    x *= np.clip(np.arange(len(x)) / (0.002 * SR), 0, 1) * np.clip((len(x) - np.arange(len(x))) / (0.01 * SR), 0, 1)
    return x / (np.abs(x).max() + 1e-9)


def riser(length=1.0):
    n = int(length * SR)
    t = np.arange(n) / SR
    p = t / length
    noise = rng.standard_normal(n + 2048)
    out = np.zeros(n + 2048)
    blk, hop = 2048, 512
    win = np.hanning(blk)
    f = np.fft.rfftfreq(blk, 1 / SR)
    for i in range(0, n, hop):
        centre = 300 * (20 ** (i / n))                   # 300 Hz -> 6 kHz
        mask = np.exp(-((np.log(np.maximum(f, 1) / centre)) ** 2) / (2 * 0.5 ** 2))
        out[i:i + blk] += np.fft.irfft(np.fft.rfft(noise[i:i + blk] * win) * mask, blk) * win
    sweep = out[:n] / (np.abs(out[:n]).max() + 1e-9)
    freq = 110 * (4 ** p)                                # rising saw underneath, 110 -> 440 Hz
    saw = 2 * ((np.cumsum(freq) / SR) % 1) - 1
    x = (0.8 * sweep + 0.25 * saw) * p ** 2.5          # crescendo, loudest at the very end
    x[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))
    return x / np.abs(x).max()


def main(events_path, duration, out_path):
    events = json.load(open(events_path))
    duration = float(duration)
    track = np.zeros((int(duration * SR) + SR, 2))
    shutter_only = np.zeros_like(track)
    for e in events:
        t, kind = e["t"], e["type"]
        pan = rng.uniform(-0.2, 0.2)
        if kind == "shutter":
            snd, pk = shutter_peak(rng.uniform(0.98, 1.02))
            g = -6 + rng.uniform(-1.5, 1)
            place_peak(track, snd, pk, t, g, pan)
            place_peak(shutter_only, snd, pk, t, g, pan)
        elif kind == "hit":
            size = e.get("size", 1.0)
            h = reverb(hit(size * 0.7), decay=0.2, wet=0.12, length=0.6)
            place_onset(track, h, t, -7 + 3 * (size - 1))
        elif kind == "flash":
            f = flash()
            place_peak(track, f, int(np.argmax(np.abs(f))), t, -9, pan)
        elif kind == "whoosh":
            w = reverb(whoosh(0.4), decay=0.2, wet=0.2, length=0.6)
            mono = np.abs(w.mean(axis=1))
            place_peak(track, w, int(np.argmax(mono[: int(0.42 * SR)])), t, -11)
        elif kind == "glitch":
            place_onset(track, glitch(), t, -13, pan)
        elif kind == "click":
            place_onset(track, click(), t, -14, pan)
        elif kind == "riser":
            r = riser(e.get("length", 1.0))
            place(track, r, t - len(r) / SR, -15)
    n = int(duration * SR)
    write_wav(out_path, track[:n])
    write_wav(out_path.replace(".wav", "_shutter_only.wav"), shutter_only[:n])
    print("sfx:", out_path, f"peak {20*np.log10(np.abs(track[:n]).max()+1e-9):.1f} dBFS, {len(events)} events")


if __name__ == "__main__":
    main(*sys.argv[1:4])
