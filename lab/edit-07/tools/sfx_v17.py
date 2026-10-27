"""Sound for the Röntgen 1895 -> WKC edit: the sfx_v12/v14 set, with an 1895 hook.

hook   frame 0: heavy hit + flash + the crack of an induction-coil spark (electrical, not glowy)
scan   under the X-ray scan: the low buzz of an induction coil / tube (electrical hum, quiet)
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


def spark(length=0.30):
    """An induction-coil spark: a sharp broadband crack with a short crackling tail."""
    n = int(length * SR)
    t = np.arange(n) / SR
    x = bandpass_noise(n, 1500, 9000) * np.exp(-t / 0.012)
    tail = bandpass_noise(n, 800, 5000) * (rng.random(n) < 0.02) * np.exp(-t / 0.08)
    x = x + 0.8 * tail
    return x / (np.abs(x).max() + 1e-9)


def coil_buzz(length):
    """The buzz of an induction coil driving an X-ray tube: 100 Hz hum with harmonics and a little crackle."""
    n = int(length * SR)
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * 100 * k * t + k) / k for k in range(1, 9))
    x += 0.15 * bandpass_noise(n, 2000, 7000) * (rng.random(n) < 0.004)
    env = np.minimum(1, t / 0.08) * np.minimum(1, (length - t) / 0.12)
    x *= env
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
            place_onset(track, spark(), max(t, 0.0005), -11)
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
        elif kind == "scan":
            place(track, coil_buzz(e.get("length", 1.0)), t, -24)
        elif kind == "riser":
            r = riser(e.get("length", 1.0))
            place(track, r, t - len(r) / SR, -15)
    n = int(duration * SR)
    write_wav(out_path, track[:n])
    write_wav(out_path.replace(".wav", "_shutter_only.wav"), shutter_only[:n])
    print("sfx:", out_path, f"peak {20*np.log10(np.abs(track[:n]).max()+1e-9):.1f} dBFS, {len(events)} events")


if __name__ == "__main__":
    main(*sys.argv[1:4])
