"""Sound effects for the Wright 1903 classy edit: ONLY the caption shutter, softer than before (client: "the same sound
effect when a new caption pops up, but softer, so it's not that dominant"). No hits, flashes, whooshes or risers:
the music carries the transitions.

Softer = about 9 dB quieter than the Prime shutter (-15 dB instead of -6) and its sharpest top end rounded off
(gentle low-pass at 7 kHz), so it reads as a quiet camera click under the jazz. Its biggest wave still lands on the
caption's first frame.
Usage: python3 sfx_v19.py EVENTS.json DURATION_S OUT.wav   (also writes OUT_shutter_only.wav for the verifier)
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
from sfx_prime import SR, place_peak, rng, shutter_peak, write_wav  # noqa: E402

GAIN_DB = -15.0
LOWPASS_HZ = 7000


def soften(x):
    """Zero-phase gentle low-pass (FFT, cosine roll-off one octave wide): the peak stays where it was."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    roll = np.clip((f - LOWPASS_HZ) / LOWPASS_HZ, 0, 1)
    X *= 0.5 + 0.5 * np.cos(np.pi * roll)
    return np.fft.irfft(X, len(x))


def main(events_path, duration, out_path):
    events = json.load(open(events_path))
    n = int(float(duration) * SR)
    track = np.zeros((n + SR, 2))
    for e in events:
        if e["type"] != "shutter":
            raise ValueError(f"unexpected event {e['type']} in the classy edit")
        snd, _ = shutter_peak(rng.uniform(0.98, 1.02))
        snd = soften(snd)
        pk = int(np.argmax(np.abs(snd)))
        place_peak(track, snd, pk, e["t"], GAIN_DB + rng.uniform(-1.0, 0.5), rng.uniform(-0.15, 0.15))
    write_wav(out_path, track[:n])
    write_wav(out_path.replace(".wav", "_shutter_only.wav"), track[:n])
    print("sfx:", out_path, f"peak {20 * np.log10(np.abs(track[:n]).max() + 1e-9):.1f} dBFS, {len(events)} soft shutters")


if __name__ == "__main__":
    main(*sys.argv[1:4])
