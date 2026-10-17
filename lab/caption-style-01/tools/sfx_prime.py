"""Synthesised caption sound effects for the Prime style, placed on a caption plan.

Sounds (all generated, so no licensing questions):
  tick     very short, soft high click on body (sans) words
  whoosh   noise swell rising into a script word (pre-roll before the word)
  hit      sub-bass drop with a short attack, audible on phone speakers
  shimmer  glassy high partials for the glossy/pink word
Writes each sound to a kit folder and a full SFX track aligned to the plan.

Usage: python3 sfx_prime.py OUT_TRACK.wav DURATION_S KIT_DIR
"""
import sys
import wave

import numpy as np

SR = 48000
rng = np.random.default_rng(7)

# (time s, class, extras) taken from the quick render's caption plan
EVENTS = [
    (0.14, "sans", {}),
    (0.94, "script", {"shimmer": True, "size": 1.0}),
    (1.49, "sans", {}), (1.80, "sans", {}), (2.16, "sans", {}), (2.43, "sans", {}), (2.73, "sans", {}),
    (2.93, "script", {"shimmer": False, "size": 0.8}),
    (3.34, "sans", {}), (3.98, "sans", {}),
    (4.27, "script", {"shimmer": True, "size": 1.25, "tail": True}),
]


def db(x):
    return 10 ** (x / 20)


def bandpass_noise(n, lo, hi):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= ((f >= lo) & (f <= hi)).astype(float) * np.exp(-((np.log(np.maximum(f, 1) / np.sqrt(lo * hi))) ** 2) / 2)
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-9)


def tick(pitch=1.0):
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    body = bandpass_noise(n, 2500 * pitch, 7000 * pitch) * np.exp(-t / 0.004)
    click = np.sin(2 * np.pi * 1900 * pitch * t) * np.exp(-t / 0.002)
    x = 0.7 * body + 0.5 * click
    return x / np.abs(x).max()


def whoosh(dur=0.32):
    """Noise through a band that sweeps upward, swelling into the hit (ends at the word)."""
    n = int(dur * SR)
    noise = rng.standard_normal(n + 2048)
    out = np.zeros(n + 2048)
    blk, hop = 2048, 512
    win = np.hanning(blk)
    f = np.fft.rfftfreq(blk, 1 / SR)
    for i in range(0, n, hop):
        p = i / n
        centre = 350 * (9 ** p)                      # 350 Hz -> ~3 kHz
        width = 1.2
        mask = np.exp(-((np.log(np.maximum(f, 1) / centre)) ** 2) / (2 * (width / 2) ** 2))
        seg = np.fft.irfft(np.fft.rfft(noise[i:i + blk] * win) * mask, blk)
        out[i:i + blk] += seg * win
    out = out[:n]
    t = np.linspace(0, 1, n)
    env = t ** 2.2 * (1 - np.exp(-(1 - t) * 60))     # ease-in swell, quick release at the end
    x = out * env
    return x / (np.abs(x).max() + 1e-9)


def hit(size=1.0):
    dur = 0.6 * size
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0, f1 = 95, 44
    freq = f1 + (f0 - f1) * np.exp(-t / 0.06)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sub = np.sin(phase) * np.exp(-t / (0.28 * size))
    sub = np.tanh(2.2 * sub) / np.tanh(2.2)          # harmonics so phones can hear it
    # phone speakers barely play below ~150 Hz: add a mid-range thud and a click on top of the sub
    thud_f = 110 + 90 * np.exp(-t / 0.03)
    thud = np.sin(2 * np.pi * np.cumsum(thud_f) / SR) * np.exp(-t / 0.09)
    knock = bandpass_noise(n, 150, 1200) * np.exp(-t / 0.015)
    click = bandpass_noise(n, 2500, 6000) * np.exp(-t / 0.004)
    x = 0.8 * sub + 0.9 * thud + 0.5 * knock + 0.35 * click
    return x / np.abs(x).max()


def shimmer(size=1.0):
    dur = 1.1
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    partials = [2093, 2637, 3136, 4186, 5274, 6272]   # C7 E7 G7 C8 E8 G8
    for k, fr in enumerate(partials):
        delay = int(rng.uniform(0, 0.07) * SR) + k * int(0.012 * SR)
        tt = t[: n - delay]
        vib = 1 + 0.002 * np.sin(2 * np.pi * 5.5 * tt)
        tone = np.sin(2 * np.pi * fr * vib * tt) * np.exp(-tt / rng.uniform(0.25, 0.5)) * (1 - np.exp(-tt / 0.003))
        x[delay:] += tone * (0.9 ** k)
    air = bandpass_noise(n, 6000, 14000) * np.exp(-t / 0.18) * 0.25
    x = x + air
    return x / np.abs(x).max()


def reverb(x, decay=0.45, length=1.4, wet=0.3):
    n = int(length * SR)
    t = np.arange(n) / SR
    irs = [rng.standard_normal(n) * np.exp(-t / decay) for _ in range(2)]
    out = []
    for ir in irs:
        ir = ir / np.sqrt((ir ** 2).sum())
        m = len(x) + n
        y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)
        out.append((1 - wet) * np.pad(x, (0, n)) + wet * y * 3)
    return np.stack(out, axis=1)


def place(track, sound, t0, gain_db, pan=0.0):
    if sound.ndim == 1:
        sound = np.stack([sound * np.sqrt(0.5 * (1 - pan)), sound * np.sqrt(0.5 * (1 + pan))], axis=1) * np.sqrt(2)
    i0 = int(round(t0 * SR))
    if i0 < 0:
        sound, i0 = sound[-i0:], 0
    end = min(len(track), i0 + len(sound))
    track[i0:end] += sound[: end - i0] * db(gain_db)


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    pcm = (x * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main(out_path, duration, kit_dir):
    duration = float(duration)
    track = np.zeros((int(duration * SR) + SR, 2))
    for i, (t, cls, ex) in enumerate(EVENTS):
        if cls == "sans":
            place(track, tick(rng.uniform(0.9, 1.12)), t, -15, pan=rng.uniform(-0.15, 0.15))
        else:
            size = ex.get("size", 1.0)
            w = whoosh(0.3)
            place(track, reverb(w, decay=0.25, wet=0.25), t - 0.3, -12 + 2 * (size - 1), pan=0.0)
            h = hit(size)
            place(track, reverb(h, decay=0.6 if ex.get("tail") else 0.3, wet=0.35 if ex.get("tail") else 0.2), t, -5 + 3 * (size - 1))
            if ex.get("shimmer"):
                place(track, reverb(shimmer(size), decay=0.5, wet=0.4), t + 0.01, -14)
    track = track[: int(duration * SR)]
    fade = int(0.12 * SR)
    track[-fade:] *= np.linspace(1, 0, fade)[:, None]
    write_wav(out_path, track)
    # the kit, one file per sound, for reuse and for review
    import os
    os.makedirs(kit_dir, exist_ok=True)
    write_wav(f"{kit_dir}/tick.wav", tick() * db(-6))
    write_wav(f"{kit_dir}/whoosh.wav", reverb(whoosh(0.3), decay=0.25, wet=0.25) * db(-3))
    write_wav(f"{kit_dir}/hit.wav", reverb(hit(1.0), wet=0.2) * db(-3))
    write_wav(f"{kit_dir}/hit_tail.wav", reverb(hit(1.25), decay=0.6, wet=0.35) * db(-3))
    write_wav(f"{kit_dir}/shimmer.wav", reverb(shimmer(), decay=0.5, wet=0.4) * db(-6))
    print("sfx track:", out_path, f"peak {20*np.log10(np.abs(track).max()+1e-9):.1f} dBFS")


if __name__ == "__main__":
    main(*sys.argv[1:4])
