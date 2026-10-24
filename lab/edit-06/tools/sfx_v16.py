"""Sound for the Galileo 1610 -> WKC edit: the sfx_v12/v14 set, with a 1610 hook.

hook   frame 0: heavy hit + flash + a brass telescope sliding out and clicking into place (mechanical, not glowy)
Every other event follows PREFERENCES.md "Sound effects per moment": shutter on every caption (its biggest wave on
the caption frame), hit on script words, flash on cuts, whoosh scene to scene, glitch on big text, click on
pop-ups, riser into a reveal.
Usage: python3 sfx_v15.py EVENTS.json DURATION_S OUT.wav
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/edit-02/tools")
from sfx_prime import SR, bandpass_noise, hit, place, place_onset, place_peak, reverb, rng, shutter_peak, whoosh, write_wav  # noqa: E402
from sfx_v12 import click, flash, glitch, riser  # noqa: E402


def telescope_slide(length=0.32):
    """A brass draw-tube sliding out: a rising band of friction noise, ending in a small metal click."""
    n = int(length * SR)
    t = np.arange(n) / SR
    lo = 700 + 2600 * (t / length)
    x = np.zeros(n)
    for k in range(0, n, 512):
        seg = bandpass_noise(min(512, n - k), lo[k], lo[k] * 1.8)
        x[k:k + len(seg)] = seg
    x *= np.sin(np.pi * np.clip(t / length, 0, 1)) ** 0.7 * 0.6
    c = click()
    c = c.mean(axis=1) if c.ndim == 2 else c
    end = n - len(c) // 3
    x[end:end + len(c)] += c[: n - end] * 1.2
    return x / (np.abs(x).max() + 1e-9)


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
            place_onset(track, reverb(hit(size * 0.7), decay=0.2, wet=0.12, length=0.6), t, -7 + 3 * (size - 1))
        elif kind == "hook":
            place_onset(track, reverb(hit(1.5), decay=0.35, wet=0.2, length=0.8), max(t, 0.0005), -3)
            place_onset(track, flash(), max(t, 0.0005), -8)
            place_onset(track, telescope_slide(), max(t, 0.0005), -12)
        elif kind == "flash":
            f = flash()
            place_peak(track, f, int(np.argmax(np.abs(f))), t, -10, pan)
        elif kind == "whoosh":
            w = reverb(whoosh(0.4), decay=0.2, wet=0.2, length=0.6)
            place_peak(track, w, int(np.argmax(np.abs(w.mean(axis=1))[: int(0.42 * SR)])), t, -11)
        elif kind == "glitch":
            place_onset(track, glitch(), max(t, 0.0005), -13, pan)
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
