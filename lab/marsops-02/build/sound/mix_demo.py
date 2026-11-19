"""Demo mix for the v2 teaser: found music (Tchaikovsky, Pathetique I, Czech National Symphony Orchestra / Musopen,
CC0) cut so its climax lands on the logo (12.0 s), real recorded effects (BigSoundBank CC0, Mixkit free licence) on
the picture's events, the client's own shutter on every caption. Master -14 LUFS, true peak under -1 dBTP.
    python -I mix_demo.py OUT.wav
"""
import json
import subprocess
import sys

import numpy as np
from scipy.signal import butter, sosfilt

M = "/home/user/SocialMediaMarketingSkill/lab/marsops-02"
SR, DUR = 48000, 15.0
N = int(SR * DUR)


def load(path, start=0.0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)] + (["-t", str(dur)] if dur else []) + ["-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()
    return x


def db(g):
    return 10 ** (g / 20)


def lp(x, hz):
    return sosfilt(butter(4, hz, "low", fs=SR, output="sos"), x, axis=0)


def hp(x, hz):
    return sosfilt(butter(2, hz, "high", fs=SR, output="sos"), x, axis=0)


def fade(x, fi=0.01, fo=0.05):
    n = len(x); a = np.ones(n, np.float32)
    k = min(n, int(fi * SR)); a[:k] = np.linspace(0, 1, k)
    k = min(n, int(fo * SR)); a[n - k:] *= np.linspace(1, 0, k)
    return x * a[:, None]


def put(bus, x, t, gain_db):
    i = int(round(t * SR))
    if i < 0: x, i = x[-i:], 0
    n = min(len(x), N - i)
    if n > 0: bus[i:i + n] += x[:n] * db(gain_db)


def steady(x, length):
    """The steadiest busy stretch of a recording (continuous pencil scratching): the window with the highest 20th
    percentile of its 50 ms loudness."""
    hop = int(0.05 * SR); e = np.sqrt((x[: len(x) // hop * hop, 0].reshape(-1, hop) ** 2).mean(1) + 1e-12)
    w = int(length / 0.05); best, bi = -1, 0
    for i in range(0, len(e) - w):
        v = np.percentile(e[i:i + w], 20)
        if v > best: best, bi = v, i
    return x[bi * hop: bi * hop + int(length * SR)]


S = f"{M}/sfx"
music = np.zeros((N, 2), np.float32); fx = np.zeros((N, 2), np.float32); log = []

# music: Pathetique I from 681.0 s, the climax (693.0) on the logo at 12.0; the quiet start lifted and gently compressed
mus = load(f"{M}/music/tchaikovsky-pathetique-i/pathetique_i_cnso-musopen.mp3", 681.0, DUR)[:N]
t = np.arange(len(mus)) / SR
lift = np.interp(t, [0, 3.0, 9.0, 12.0, 15.0], [7.0, 6.0, 2.0, 0.0, 0.0])          # dB: brings the start forward
mus = mus * db(lift)[:, None]
env = np.sqrt(np.convolve((mus ** 2).mean(1), np.ones(int(0.08 * SR)) / int(0.08 * SR), "same") + 1e-12)
lvl = 20 * np.log10(env + 1e-9); thr = -20.0
gr = np.where(lvl > thr, (lvl - thr) * (1 - 1 / 2.2), 0.0)                          # 2.2:1 above -20 dBFS
mus = mus * db(-gr)[:, None]
fo = np.clip((15.0 - t) / 0.9, 0, 1); mus = mus * (fo ** 1.5)[:, None]                # out over the last 0.9 s
music[: len(mus)] += mus

# the page: a big paper page settles as the film opens
put(fx, fade(load(f"{S}/mixkit-1105-big-paper-page-turn/mixkit-1105-big-paper-page-turn.wav"), 0.002, 0.1), 0.0, -9)
# pencil: the circle and the construction lines (0.25-2.4), the globe lines (3.75-4.9), the shield drawn (7.35-8.65)
pen = load(f"{S}/bsb-pencil-s0221/pencil-s0221.wav")
put(fx, fade(hp(steady(pen, 2.3), 300), 0.08, 0.35), 0.22, -13)
put(fx, fade(hp(steady(pen[int(12 * SR):], 1.3), 300), 0.1, 0.3), 3.75, -19)
put(fx, fade(hp(steady(pen[int(20 * SR):], 1.4), 300), 0.06, 0.3), 7.32, -15)
# the paper turns to blueprint: a large sheet handled
rm = load(f"{S}/bsb-road-map-4-s2799/road-map-4-s2799.wav")
put(fx, fade(hp(steady(rm, 0.7), 200), 0.05, 0.25), 2.95, -17)
# space: a wind swell under Mars and the Sun
put(fx, fade(lp(load(f"{S}/mixkit-1170-cinematic-wind-swell/mixkit-1170-cinematic-wind-swell.wav"), 6000), 0.6, 1.2), 3.3, -21)
# the flare: a deep, distant impact
put(fx, fade(lp(load(f"{S}/mixkit-1686-underground-explosion-impact-echo/mixkit-1686-underground-explosion-impact-echo.wav"), 2500), 0.003, 0.6), cues_flare := 6.25, -15)
# the shield closes: a short beep
put(fx, fade(load(f"{S}/bsb-aerospace-beep-1-s2380/aerospace-beep-1-s2380.wav"), 0.002, 0.04), 8.65, -27)
# into the logo: riser, then the impact on 12.0 (the music's peak)
ris = load(f"{S}/mixkit-790-cinematic-trailer-riser/mixkit-790-cinematic-trailer-riser.wav")
put(fx, fade(ris, 0.3, 0.05), 12.0 - len(ris) / SR + 0.05, -17)
imp = load(f"{S}/mixkit-788-big-cinematic-impact/mixkit-788-big-cinematic-impact.mp3")
pk = int(np.argmax(np.abs(imp[: int(1.0 * SR), 0])))
put(fx, fade(imp, 0.001, 1.5), 12.0 - pk / SR, -14)
# the client's shutter on every caption: its biggest wave on the caption's first frame; softened
sh = load(f"{S}/client_camera_shutter.wav"); spk = int(np.argmax(np.abs(sh).max(1)))
sh = lp(sh, 7000)
for k, tc in enumerate([0.45, 1.65, 3.55, 5.20, 7.10, 9.30]):
    put(fx, sh * (1 + 0.06 * ((-1) ** k)), tc - spk / SR, -15)

mix = music + fx
tmp = sys.argv[1] + ".pre.wav"
import wave
pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
with wave.open(tmp, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
# master: two-pass loudnorm to -14 LUFS, true peak -1 dBTP
m = subprocess.run(["ffmpeg", "-v", "info", "-i", tmp, "-af", "loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(m[m.rindex("{"): m.rindex("}") + 1])
af = (f"loudnorm=I=-14:TP=-1.0:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
      f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,aresample=48000")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", af, "-ar", "48000", sys.argv[1]], check=True)
print("pre-master", j["input_i"], "LUFS, TP", j["input_tp"], "; shutter peak at", round(spk / SR * 1000, 1), "ms")
