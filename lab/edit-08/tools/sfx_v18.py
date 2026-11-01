"""Sound for the Savery 1698 -> WKC edit: the sfx_v12/v14 set, with a steam hook.

hook   frame 0: heavy hit + flash + a burst of steam from a pressure valve (mechanical, not glowy)
hiss   in the engine animation: steam filling the vessel / pushing the water out (quiet)
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


def steam_hiss(length=0.6, burst=False):
    """Steam escaping: band-passed noise, a fast attack (a burst for the hook), a decaying tail, slight flutter."""
    n = int(length * SR)
    t = np.arange(n) / SR
    x = bandpass_noise(n, 1800, 9500) + 0.35 * bandpass_noise(n, 400, 1800)
    flutter = 1 + 0.12 * np.sin(2 * np.pi * 23 * t)
    env = (1 - np.exp(-t / (0.006 if burst else 0.05))) * np.exp(-t / (0.22 if burst else length * 0.6))
    x = x * flutter * env
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
            place_onset(track, steam_hiss(0.8, burst=True), max(t, 0.0005), -11)
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
        elif kind == "hiss":
            place_onset(track, steam_hiss(e.get("length", 0.6)), t, -20)
        elif kind == "riser":
            r = riser(e.get("length", 1.0))
            place(track, r, t - len(r) / SR, -15)
    n = int(duration * SR)
    write_wav(out_path, track[:n])
    write_wav(out_path.replace(".wav", "_shutter_only.wav"), shutter_only[:n])
    print("sfx:", out_path, f"peak {20*np.log10(np.abs(track[:n]).max()+1e-9):.1f} dBFS, {len(events)} events")


if __name__ == "__main__":
    main(*sys.argv[1:4])
