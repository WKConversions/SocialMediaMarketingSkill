"""Synthesised caption sound effects for the Prime style, placed on a caption plan.

Sounds (all generated, so no licensing questions). Client rule: never any magical, glowy,
sparkle or chime sounds; keep everything mechanical/physical (camera clicks, whooshes, hits).

  tick     very short, soft high click on body (sans) words
  whoosh   noise swell rising into a script word (pre-roll before the word)
  hit      sub-bass drop with a short attack, audible on phone speakers
Writes each sound to a kit folder and a full SFX track aligned to the plan.

Usage: python3 sfx_prime.py OUT_TRACK.wav DURATION_S KIT_DIR [PLAN.json]
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
    (0.14, "user_full", {}),                                       # dit is een (opening)
    (0.94, "script", {"burst": (4, 0.07, 0.035), "size": 1.0, "whoosh": True}),    # Testvideo: accelerating
    (1.49, None, {}),                                              # om te: silent
    (1.80, "user_click", {}),                                      # kijken
    (2.16, None, {}),                                              # of jij: silent
    (2.43, "user_full", {"pitch": 0.96}),                          # goed
    (2.73, None, {}),                                              # kan: silent
    (2.93, "script", {"burst": (3, 0.055, 0.04), "size": 0.85}),   # Editen
    (3.34, "user_double", {}),                                     # wat ik
    (3.98, "user_click", {"pitch": 1.04}),                         # graag zou
    (4.27, "script", {"burst": (5, 0.03, 0.07), "size": 1.25, "tail": True, "whoosh": True}),  # Willen: decelerating
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
    x *= np.exp(-t / (0.12 * size)) ** 0.5 * np.clip((dur - t) / 0.08, 0, 1)   # fade fully out: no cut-off click
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


# ---- the client's own camera shutter (assets/sfx/camera_shutter.wav) is the caption sound ----------

SHUTTER_PATH = "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/assets/sfx/camera_shutter.wav"
SHUTTER_SYNC_S = 0.192      # loudest click inside the file: this point lands on the caption frame


def _load_shutter():
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", SHUTTER_PATH, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(float)
    a, b = int(0.095 * SR), int(0.30 * SR)            # trim the silence around the ~100 ms of clicks
    x = x[a:b].copy()
    x[: int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))
    x[-int(0.03 * SR):] *= np.linspace(1, 0, int(0.03 * SR))
    return x / np.abs(x).max(), SHUTTER_SYNC_S - 0.095


_SHUTTER, _SHUTTER_SYNC = _load_shutter()


def _repitch(x, factor):
    idx = np.arange(0, len(x) - 1, factor)
    return np.interp(idx, np.arange(len(x)), x)


def shutter_peak(p=1.0):
    """The full shutter (all clicks). Returns (sound, index of its biggest wave). Client rule: that
    biggest wave is the point that lands on the first frame of a caption."""
    snd = _repitch(_SHUTTER, p)
    return snd, int(np.argmax(np.abs(snd)))


def place_peak(track, sound, peak_idx, t, gain_db, pan=0.0):
    """Place a sound so that sample `peak_idx` lands exactly at time t."""
    place(track, sound, t - peak_idx / SR, gain_db, pan)


def user_full(p=1.0):
    """The whole shutter. Returns (sound, sync offset in s)."""
    return _repitch(_SHUTTER, p), _SHUTTER_SYNC / p


def user_click(p=1.0):
    """Only the final, loudest click of the shutter."""
    a = int((0.185 - 0.095) * SR)
    x = _SHUTTER[a:].copy()
    x[: int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    return _repitch(x / np.abs(x).max(), p), 0.012 / p


def user_double(p=1.0):
    one, sync = user_full(p)
    two, _ = user_click(p * 1.03)
    gap = int(rng.uniform(0.075, 0.095) * SR)
    out = np.zeros(max(len(one), gap + len(two)))
    out[: len(one)] += one
    out[gap: gap + len(two)] += two * db(-5)
    return out / np.abs(out).max(), sync


def user_stutter(n=4, first_gap=0.06, last_gap=0.03, p=1.0):
    """A burst built from the client's shutter clicks; returns (sound, time of the last click)."""
    clicks, t = [], 0.0
    for k in range(n):
        snd, sync = (user_click if k % 2 else user_full)(p * rng.uniform(0.96, 1.05))
        clicks.append((t - sync, snd, rng.uniform(-3, 0)))
        if k < n - 1:
            t += first_gap + (last_gap - first_gap) * k / max(1, n - 2)
    lead = -min(c[0] for c in clicks)
    end = max(c[0] + lead + len(c[1]) / SR for c in clicks)
    out = np.zeros(int(end * SR) + 1)
    for t0, snd, g in clicks:
        i = int((t0 + lead) * SR)
        out[i:i + len(snd)] += snd * db(g)
    return out / np.abs(out).max(), t + lead


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


FPS = 30


def onset(sound, rel_db=-30):
    """Index of the first sample within rel_db of the sound's peak."""
    mono = np.abs(sound if sound.ndim == 1 else sound.mean(axis=1))
    return int(np.argmax(mono > mono.max() * db(rel_db)))


def place_onset(track, sound, t, gain_db, pan=0.0):
    """Place a sound so that its first audible sample lands exactly at time t."""
    place(track, sound, t - onset(sound) / SR, gain_db, pan)


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
    # Client rules: a sound only where a new caption appears, and the biggest wave of the shutter lands
    # exactly on the first frame of that caption. Script words add a short hit starting on that frame.
    shutter_only = np.zeros_like(track)
    for t, kind, ex in EVENTS:
        if kind is None:
            continue
        frame_t = np.ceil(t * FPS - 1e-6) / FPS            # the caption becomes visible on this frame
        pan = rng.uniform(-0.2, 0.2)
        gain = -6 + rng.uniform(-1.5, 1)
        snd, pk = shutter_peak(ex.get("pitch", 1.0) * rng.uniform(0.98, 1.02))
        place_peak(track, snd, pk, frame_t, gain, pan)
        place_peak(shutter_only, snd, pk, frame_t, gain, pan)
        if kind == "script":
            size = ex.get("size", 1.0)
            h = reverb(hit(size * 0.7), decay=0.25 if ex.get("tail") else 0.12, wet=0.18 if ex.get("tail") else 0.08, length=0.6)
            place_onset(track, h, frame_t, -6 + 3 * (size - 1))
    write_wav(out_path.replace(".wav", "_shutter_only.wav"), shutter_only[: int(duration * SR)])
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
    write_wav(f"{kit_dir}/client_shutter_click.wav", user_click()[0] * db(-4))
    write_wav(f"{kit_dir}/client_shutter_double.wav", user_double()[0] * db(-4))
    write_wav(f"{kit_dir}/client_shutter_stutter.wav", user_stutter(4, 0.07, 0.035)[0] * db(-4))
    write_wav(f"{kit_dir}/stutter_accel.wav", reverb(stutter(5, 0.055, 0.022), decay=0.15, wet=0.15) * db(-4))
    write_wav(f"{kit_dir}/stutter_decel.wav", reverb(stutter(6, 0.02, 0.06), decay=0.15, wet=0.15) * db(-4))
    write_wav(f"{kit_dir}/whoosh.wav", reverb(whoosh(0.3), decay=0.25, wet=0.25) * db(-3))
    write_wav(f"{kit_dir}/hit.wav", reverb(hit(1.0), wet=0.2) * db(-3))
    write_wav(f"{kit_dir}/hit_tail.wav", reverb(hit(1.25), decay=0.6, wet=0.35) * db(-3))
    print("sfx track:", out_path, f"peak {20*np.log10(np.abs(track).max()+1e-9):.1f} dBFS")


def load_events(path):
    """Use the same plan file as the caption renderer, so sounds always sit on the caption frames."""
    import json
    return [(p["start"], p["sfx"], p.get("sfx_opts", {})) for p in json.load(open(path))]


if __name__ == "__main__":
    if len(sys.argv) > 4:
        EVENTS[:] = load_events(sys.argv[4])
    main(*sys.argv[1:4])
