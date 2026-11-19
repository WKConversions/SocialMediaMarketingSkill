"""Mix for the v4 showcase teaser: found minimal music low in the background, subtle modern effects on the film's events
(cues from render4.html), master -16 LUFS, true peak under -1 dBTP.
    python -I mix_v4.py OUT.wav
Music: "What it Takes" (Eugenio Mininni, Mixkit Stock Music Free License) from 121.5 s: flat and minimal, a soft
entry at 12.0 s (the particle logo forms). Effects: Mixkit (free licence), Kenney (CC0), a real Geiger counter
(Internet Archive, Public Domain Mark).
"""
import json
import subprocess
import sys
import wave

import numpy as np
from scipy.signal import butter, sosfilt

SR, DUR = 48000, 15.0
N = int(SR * DUR)
OUT = sys.argv[1]
M = "/home/user/SocialMediaMarketingSkill/lab/marsops-03"
FX = f"{M}/sfx"


def load(path, start=0.0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)] + (["-t", str(dur)] if dur else []) + ["-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()


def db(g):
    return 10 ** (g / 20)


def filt(x, hz, kind):
    return sosfilt(butter(2 if kind == "high" else 4, hz, kind, fs=SR, output="sos"), x, axis=0)


def fade(x, fi=0.01, fo=0.05):
    n = len(x); a = np.ones(n, np.float32)
    k = min(n, int(fi * SR)); a[:k] = np.linspace(0, 1, k)
    k = min(n, int(fo * SR)); a[n - k:] *= np.linspace(1, 0, k)
    return x * a[:, None]


def put(bus, x, t, g):
    i = int(round(t * SR))
    if i < 0: x, i = x[-i:], 0
    n = min(len(x), N - i)
    if n > 0: bus[i:i + n] += x[:n] * db(g)


def peak(x, within=None):
    w = len(x) if within is None else min(len(x), int(within * SR))
    return int(np.argmax(np.abs(x[:w]).max(1))) / SR


def env(points):
    t = np.arange(N) / SR
    return db(np.interp(t, [p[0] for p in points], [p[1] for p in points]))[:, None]


music = np.zeros((N, 2), np.float32); fx = np.zeros((N, 2), np.float32)
f = lambda d: f"{FX}/{d}"

# music, low; a little more presence as the logo forms; out over the last 0.9 s
mus = load(f"{M}/music/what-it-takes-mixkit-616/what-it-takes-mixkit-616.mp3", 121.5, DUR)[:N]
music[: len(mus)] += mus
music *= env([(0, -1.5), (1.5, 0), (11.4, 0), (12.3, 1.0), (14.1, 1.0), (15.0, -30)])

# a dark room tone under everything (space), very low
bed = filt(load(f("mixkit-3081-dark-cinematic-room-tone/mixkit-3081-dark-cinematic-room-tone.wav"), 10.0, DUR), 4000, "low")[:N]
put(fx, fade(bed, 1.2, 1.2), 0, -30)

# air: the crane down onto Mars, the whip to the astronaut, the pull back to the station
w1 = load(f("mixkit-1489-air-woosh/mixkit-1489-air-woosh.wav")); put(fx, fade(w1, 0.3, 0.3), 1.55 - peak(w1), -24)
w2 = load(f("mixkit-2632-whoosh-wind-sweep/mixkit-2632-whoosh-wind-sweep.wav")); put(fx, fade(w2, 0.02, 0.2), 6.2 - peak(w2), -19)
w3 = load(f("mixkit-168-fast-air-sweep-transition/mixkit-168-fast-air-sweep-transition.wav")); put(fx, fade(w3, 0.02, 0.15), 9.7 - peak(w3), -24)

# interface: a soft tick on each headline and on each callout; the HUD drawing in
tick = filt(load(f("mixkit-3124-modern-technology-select/mixkit-3124-modern-technology-select.wav")), 9000, "low")
for k, tc in enumerate([0.55, 3.15, 6.35]):
    put(fx, tick, tc - peak(tick, 0.2), -25)
kt = load(f("kenney-interface-sounds-tick-002/tick_002.ogg"))
for k, tc in enumerate([9.35, 9.75, 10.15]):
    put(fx, kt, tc + 0.35, -23)
for k in range(10):                                    # the HUD's ticks being drawn around Mars (3.0-3.9)
    put(fx, kt * (0.6 + 0.4 * (k % 2)), 3.0 + k * 0.09, -33)

# radiation: a real Geiger counter while the wind blows unprotected; it falls away as the shield closes (8.2)
gg = filt(load(f("ia-teoli-geiger-4/teoli-geiger-counter-sounds-4.mp3")), 200, "high")
g1 = fade(gg[int(1.0 * SR): int(1.0 * SR) + int(5.4 * SR)], 0.5, 0.9) * env([(0, 0)])[: int(5.4 * SR)]
put(fx, g1, 3.0, -25)

# the hologram powering on (glitches) and the scan sweeping up the astronaut
cr = filt(load(f("mixkit-1455-static-electric-crackles/mixkit-1455-static-electric-crackles.wav")), 300, "high")
for tg, g in [(6.05, -24), (8.92, -27), (11.5, -27)]:
    put(fx, fade(cr[: int(0.35 * SR)], 0.005, 0.12), tg, g)
sw = load(f("mixkit-2522-sci-fi-sweep/mixkit-2522-sci-fi-sweep.wav")); put(fx, fade(sw, 0.05, 0.3), 6.15, -24)

# the shield building and closing (7.4-8.3): a force field, a bass swell under it
ff = load(f("kenney-sci-fi-sounds-forceField-001/forceField_001.ogg")); put(fx, fade(ff, 0.01, 0.3), 8.25 - peak(ff), -21)
bs = load(f("mixkit-2668-simple-swell-bass/mixkit-2668-simple-swell-bass.wav")) if False else load(f("mixkit-2666-bass-double-swell/mixkit-2666-bass-double-swell.wav"))
put(fx, fade(bs[: int(2.2 * SR)], 0.4, 0.6), 7.3, -24)

# the points streaming into the logo, and the soft hit as it locks (12.55); the music's entry is at 12.0
w4 = load(f("mixkit-1489-air-woosh/mixkit-1489-air-woosh.wav")); put(fx, fade(w4, 0.4, 0.3), 12.4 - peak(w4), -24)
hit = filt(load(f("mixkit-2913-epic-futuristic-movie-accent/mixkit-2913-epic-futuristic-movie-accent.wav")), 6000, "low")
put(fx, fade(hit, 0.002, 1.4), 12.55 - peak(hit, 1.0), -18)

mix = music * db(-4) + fx * db(6)          # the music in the background (the client's note), the effects present but subtle
tmp = OUT + ".pre.wav"
with wave.open(tmp, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
m = subprocess.run(["ffmpeg", "-v", "info", "-i", tmp, "-af", "loudnorm=I=-16:TP=-1.0:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(m[m.rindex("{"): m.rindex("}") + 1])
af = (f"loudnorm=I=-16:TP=-1.0:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
      f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,aresample=48000")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", af, "-ar", "48000", OUT], check=True)
for name, bus in (("music", music), ("fx", fx)):
    with wave.open(OUT + f".{name}.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(bus, -1, 1) * 32767).astype(np.int16).tobytes())
print("pre-master", j["input_i"], "LUFS")
