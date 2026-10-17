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
# Hand-placed so it never becomes a pattern: about a third of the word changes stay silent, no sound type
# twice in a row, accents on phrase starts and the script words. (time s, kind, options)
EVENTS = [
    (0.14, "shutter_dslr", {}),                                   # dit is een (opening)
    (0.94, "script", {"burst": (5, 0.055, 0.022), "shimmer": True, "size": 1.0, "whoosh": True}),  # Testvideo
    (1.49, None, {}),                                             # om te: silent
    (1.80, "shutter_mirrorless", {}),                             # kijken
    (2.16, None, {}),                                             # of jij: silent
    (2.43, "film_ratchet", {}),                                   # goed
    (2.73, None, {}),                                             # kan: silent
    (2.93, "script", {"burst": (3, 0.045, 0.03), "shimmer": False, "size": 0.85}),  # Editen
    (3.34, "shutter_double", {}),                                 # wat ik
    (3.98, "shutter_mirrorless", {"pitch": 0.9}),                 # graag zou
    (4.27, "script", {"burst": (6, 0.02, 0.06), "shimmer": True, "size": 1.25, "tail": True, "whoosh": True}),  # Willen: decelerating
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


# ---- camera family: shutters, clicks, ratchets, stutter bursts -------------------------------

def mech_click(pitch=1.0, bright=1.0, decay=1.0):
    """One mechanical click: a bright noise snap, a few short metal resonances and a small body thunk."""
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    snap = bandpass_noise(n, 1800 * pitch, 9000 * min(1.6, pitch * bright)) * np.exp(-t / (0.0022 * decay))
    ring = sum(a * np.sin(2 * np.pi * f * pitch * t) * np.exp(-t / (d * decay))
               for f, a, d in [(1150, 0.5, 0.009), (2650, 0.35, 0.006), (4300, 0.25 * bright, 0.004)])
    body = np.sin(2 * np.pi * 190 * pitch * t) * np.exp(-t / (0.008 * decay)) * 0.5
    x = snap + ring + body
    return x / np.abs(x).max()


def seq(clicks):
    """clicks: list of (time s, sound, gain dB) -> one mono buffer."""
    end = max(c[0] + len(c[1]) / SR for c in clicks)
    out = np.zeros(int(end * SR) + 1)
    for t0, snd, g in clicks:
        i = int(t0 * SR)
        out[i:i + len(snd)] += snd * db(g)
    return out / np.abs(out).max()


def shutter_dslr(p=1.0):
    gap = rng.uniform(0.038, 0.055)                     # mirror up ... shutter/mirror down
    return seq([(0, mech_click(p * 0.85, 0.8, 1.3), 0), (gap, mech_click(p * 1.1, 1.0, 1.0), -3)])


def shutter_mirrorless(p=1.0):
    return mech_click(p * 1.25, 1.3, 0.7)


def shutter_double(p=1.0):
    g = rng.uniform(0.07, 0.1)
    return seq([(0, shutter_dslr(p), 0), (g, shutter_dslr(p * 1.04), -4)])


def film_ratchet(p=1.0, teeth=5):
    clicks, t = [], 0.0
    for k in range(teeth):
        clicks.append((t, mech_click(p * rng.uniform(1.3, 1.5), 0.7, 0.5), -2 * k * 0.6 + rng.uniform(-1.5, 0)))
        t += rng.uniform(0.016, 0.024)
    return seq(clicks)


def stutter(n=5, first_gap=0.06, last_gap=0.025, p=1.0):
    """Continuous-shooting burst; gaps go from first_gap to last_gap (accelerating if first > last)."""
    clicks, t = [], 0.0
    for k in range(n):
        clicks.append((t, shutter_mirrorless(p * rng.uniform(0.95, 1.06)) if k % 2 else mech_click(p * rng.uniform(0.9, 1.05)),
                       rng.uniform(-3, 0)))
        if k < n - 1:
            t += first_gap + (last_gap - first_gap) * k / max(1, n - 2)
    return seq(clicks)


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
    makers = {"shutter_dslr": shutter_dslr, "shutter_mirrorless": shutter_mirrorless,
              "shutter_double": shutter_double, "film_ratchet": film_ratchet}
    for t, kind, ex in EVENTS:
        if kind is None:
            continue
        pan = rng.uniform(-0.3, 0.3)
        if kind != "script":
            snd = makers[kind](ex.get("pitch", 1.0) * rng.uniform(0.94, 1.06))
            place(track, reverb(snd, decay=0.12, wet=0.12), t + rng.uniform(-0.008, 0.004), -9 + rng.uniform(-2, 1.5), pan)
            continue
        size = ex.get("size", 1.0)
        n, g0, g1 = ex["burst"]
        burst = stutter(n, g0, g1, p=rng.uniform(0.95, 1.05))
        place(track, reverb(burst, decay=0.15, wet=0.15), t - len(burst) / SR + 0.012, -10, pan)   # burst lands on the word
        if ex.get("whoosh"):
            place(track, reverb(whoosh(0.3), decay=0.25, wet=0.25), t - 0.3, -17)
        place(track, reverb(hit(size), decay=0.6 if ex.get("tail") else 0.3, wet=0.35 if ex.get("tail") else 0.2), t, -6 + 3 * (size - 1))
        if ex.get("shimmer"):
            place(track, reverb(shimmer(size), decay=0.5, wet=0.4), t + 0.01, -16)
    track = track[: int(duration * SR)]
    fade = int(0.12 * SR)
    track[-fade:] *= np.linspace(1, 0, fade)[:, None]
    write_wav(out_path, track)
    # the kit, one file per sound, for reuse and for review
    import os
    os.makedirs(kit_dir, exist_ok=True)
    write_wav(f"{kit_dir}/tick.wav", tick() * db(-6))
    for name, fn in [("shutter_dslr", shutter_dslr), ("shutter_mirrorless", shutter_mirrorless),
                     ("shutter_double", shutter_double), ("film_ratchet", film_ratchet)]:
        write_wav(f"{kit_dir}/{name}.wav", reverb(fn(), decay=0.12, wet=0.12) * db(-4))
    write_wav(f"{kit_dir}/stutter_accel.wav", reverb(stutter(5, 0.055, 0.022), decay=0.15, wet=0.15) * db(-4))
    write_wav(f"{kit_dir}/stutter_decel.wav", reverb(stutter(6, 0.02, 0.06), decay=0.15, wet=0.15) * db(-4))
    write_wav(f"{kit_dir}/whoosh.wav", reverb(whoosh(0.3), decay=0.25, wet=0.25) * db(-3))
    write_wav(f"{kit_dir}/hit.wav", reverb(hit(1.0), wet=0.2) * db(-3))
    write_wav(f"{kit_dir}/hit_tail.wav", reverb(hit(1.25), decay=0.6, wet=0.35) * db(-3))
    write_wav(f"{kit_dir}/shimmer.wav", reverb(shimmer(), decay=0.5, wet=0.4) * db(-6))
    print("sfx track:", out_path, f"peak {20*np.log10(np.abs(track).max()+1e-9):.1f} dBFS")


if __name__ == "__main__":
    main(*sys.argv[1:4])
