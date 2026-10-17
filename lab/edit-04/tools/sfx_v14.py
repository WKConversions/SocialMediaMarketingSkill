"""Sound for the Brooklyn Bridge -> WKC edit: the sfx_v12 set plus a hook and a film-projector bed.

hook       frame 0: heavy hit + flash + a projector motor clattering up to speed (mechanical, not glowy)
projector  quiet sprocket rattle under every archival insert, with short fades
Usage: python3 sfx_v14.py EVENTS.json DURATION_S OUT.wav
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/edit-02/tools")
from sfx_prime import SR, bandpass_noise, db, hit, place, place_onset, place_peak, reverb, rng, shutter_peak, whoosh, write_wav  # noqa: E402
from sfx_v12 import click, flash, glitch, riser  # noqa: E402


def sprocket(rate, n, level_curve=None):
    """Projector clatter: a train of short clicks at `rate` per second over n samples (rate may be an array)."""
    rate = np.broadcast_to(np.asarray(rate, float), (n,))
    phase = np.cumsum(rate / SR)
    ticks = np.where(np.diff(np.floor(phase), prepend=0) > 0)[0]
    out = np.zeros(n)
    tick = bandpass_noise(int(0.012 * SR), 900, 6000) * np.exp(-np.arange(int(0.012 * SR)) / SR / 0.0025)
    for k in ticks:
        e = min(n, k + len(tick))
        out[k:e] += tick[: e - k] * rng.uniform(0.7, 1.0)
    hum = bandpass_noise(n, 80, 400) * 0.15
    x = out + hum
    if level_curve is not None:
        x *= level_curve
    return x / (np.abs(x).max() + 1e-9)


def hook():
    n = int(0.7 * SR)
    t = np.arange(n) / SR
    spin = sprocket(6 + 18 * (1 - np.exp(-t / 0.18)), n, np.exp(-t / 0.35))      # clatter spinning up, then fading
    return spin


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
            f = flash()
            place_onset(track, f, max(t, 0.0005), -8)
            place_onset(track, hook(), max(t, 0.0005), -14)
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
        elif kind == "projector":
            n = int((e["end"] - t) * SR)
            fade = np.ones(n)
            m = min(n // 2, int(0.03 * SR))
            fade[:m] = np.linspace(0, 1, m)
            fade[-m:] = np.linspace(1, 0, m)
            place(track, sprocket(24, n) * fade, t, -26)
    n = int(duration * SR)
    write_wav(out_path, track[:n])
    write_wav(out_path.replace(".wav", "_shutter_only.wav"), shutter_only[:n])
    print("sfx:", out_path, f"peak {20*np.log10(np.abs(track[:n]).max()+1e-9):.1f} dBFS, {len(events)} events")


if __name__ == "__main__":
    main(*sys.argv[1:4])
