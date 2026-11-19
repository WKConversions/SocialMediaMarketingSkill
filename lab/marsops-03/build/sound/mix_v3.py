"""Mix for the v3 teaser: found minimal music low in the background, subtle modern effects on the picture's events
(cues from the render page), master about -16 LUFS, true peak under -1 dBTP.
    python -I mix_v3.py picks.json OUT.wav
picks.json: {"music": {"file", "start", "gain_db"}, "whoosh": [file...], "tick": file, "radiation": file,
             "swell": file, "hit": file, "bed": file|null}
"""
import json
import subprocess
import sys
import wave

import numpy as np
from scipy.signal import butter, sosfilt

SR, DUR = 48000, 15.0
N = int(SR * DUR)
P = json.load(open(sys.argv[1]))
OUT = sys.argv[2]


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


def peak_at(x, within=1.0):
    return int(np.argmax(np.abs(x[: int(within * SR)]).max(1)))


def env(points):
    """A gain curve (dB) through (time, dB) points."""
    t = np.arange(N) / SR
    return db(np.interp(t, [p[0] for p in points], [p[1] for p in points]))[:, None]


music = np.zeros((N, 2), np.float32); fx = np.zeros((N, 2), np.float32)

# music: the chosen window, low; a gentle rise into the logo, out over the last second
m = P["music"]
mus = load(m["file"], m["start"], DUR)[:N]
music[: len(mus)] += mus * db(m.get("gain_db", 0))
music *= env([(0, -2), (2.5, 0), (11.0, 0), (12.2, 1.5), (14.0, 1.5), (15.0, -30)])

# air on the three changes (Mars pushes in, the card rises, the base)
for k, (t_peak, g) in enumerate([(3.0, -20), (6.35, -21), (9.65, -22)]):
    w = load(P["whoosh"][k % len(P["whoosh"])])
    put(fx, fade(w, 0.02, 0.15), t_peak - peak_at(w, len(w) / SR) / SR, g)
# a soft tick on each line of text
tick = filt(load(P["tick"]), 9000, "low")
for k, tc in enumerate([0.35, 3.35, 6.75, 10.1, 10.35, 10.6]):
    put(fx, tick * (1 + 0.05 * ((-1) ** k)), tc - peak_at(tick, 0.2) / SR, -27)
# radiation under the Sun's beat and while the card is unprotected; it stops when the shield closes (8.45)
rad = filt(load(P["radiation"]), 180, "high")
seg = lambda a, b: rad[int(a * SR) % max(1, len(rad) - int((b - a) * SR) - 1):][: int((b - a) * SR)]
put(fx, fade(seg(0, 2.9), 0.4, 0.5), 3.45, -24)
put(fx, fade(seg(0.7, 1.55), 0.25, 0.08), 6.95, -25)
# the shield forming: a soft swell peaking as it closes
sw = load(P["swell"])
put(fx, fade(sw, 0.3, 0.4), 8.45 - peak_at(sw, len(sw) / SR) / SR, -22)
# the logo: a soft low hit
hit = filt(load(P["hit"]), 3500, "low")
put(fx, fade(hit, 0.002, 1.2), 11.95 - peak_at(hit, 1.0) / SR, -16)
if P.get("bed"):
    bed = filt(load(P["bed"], 0, DUR), 5000, "low")[:N]
    put(fx, fade(bed, 1.0, 1.5), 0, -32)

mix = music + fx
tmp = OUT + ".pre.wav"
with wave.open(tmp, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
m = subprocess.run(["ffmpeg", "-v", "info", "-i", tmp, "-af", "loudnorm=I=-16:TP=-1.0:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(m[m.rindex("{"): m.rindex("}") + 1])
af = (f"loudnorm=I=-16:TP=-1.0:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
      f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,aresample=48000")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", af, "-ar", "48000", OUT], check=True)
# stems, for the report: music and effects loudness before the master
for name, bus in (("music", music), ("fx", fx)):
    with wave.open(OUT + f".{name}.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(bus, -1, 1) * 32767).astype(np.int16).tobytes())
print("pre-master", j["input_i"], "LUFS")
