"""Documentary mix: the client's narration (starts 1.5 s), calm music under it ("Valley Sunset", Mixkit free licence),
a space room tone, soft real/recorded effects on the cuts. Master -16 LUFS, TP -1 dBTP.  python -I mix5.py OUT.wav"""
import json, subprocess, sys, wave
import numpy as np
from scipy.signal import butter, sosfilt
SR, DUR = 48000, 46.0; N = int(SR * DUR); OUT = sys.argv[1]
M3 = "/home/user/SocialMediaMarketingSkill/lab/marsops-03"; FX = M3 + "/sfx"
def load(p, s=0.0, d=None):
    c = ["ffmpeg", "-v", "error", "-ss", str(s)] + (["-t", str(d)] if d else []) + ["-i", p, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(c, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()
db = lambda g: 10 ** (g / 20)
lp = lambda x, hz: sosfilt(butter(4, hz, "low", fs=SR, output="sos"), x, axis=0)
def fade(x, a=0.01, b=0.05):
    n = len(x); e = np.ones(n, np.float32); k = min(n, int(a * SR)); e[:k] = np.linspace(0, 1, k); k = min(n, int(b * SR)); e[n - k:] *= np.linspace(1, 0, k); return x * e[:, None]
def put(bus, x, t, g):
    i = int(round(t * SR)); x = x[-i:] if i < 0 else x; i = max(i, 0); n = min(len(x), N - i)
    if n > 0: bus[i:i + n] += x[:n] * db(g)
pk = lambda x: int(np.argmax(np.abs(x).max(1))) / SR
vo = np.zeros((N, 2), np.float32); mu = np.zeros((N, 2), np.float32); fx = np.zeros((N, 2), np.float32)
v = load("/home/user/SocialMediaMarketingSkill/lab/marsops-04/vo/vo.mp3"); put(vo, v, 1.5, 0)
m = load(M3 + "/music/valley-sunset-mixkit-127/valley-sunset-mixkit-127.mp3", 6.0, DUR)[:N]; mu[:len(m)] += m
# duck the music under the voice (envelope of the narration), up in the pauses and after the last word
e = np.abs(vo).max(1); win = int(0.35 * SR); env = np.convolve(e, np.ones(win) / win, "same"); env = np.convolve((env > 0.01).astype(float), np.ones(int(0.6 * SR)) / int(0.6 * SR), "same")
t = np.arange(N) / SR
mu *= (db(-11 * env) * np.clip(t / 1.0, 0, 1) * np.clip((DUR - t) / 2.5, 0, 1))[:, None]
bed = lp(load(FX + "/mixkit-3081-dark-cinematic-room-tone/mixkit-3081-dark-cinematic-room-tone.wav", 0, DUR), 3500)[:N]; put(fx, fade(bed, 1.5, 2.5), 0, -26)
w = load(FX + "/mixkit-1489-air-woosh/mixkit-1489-air-woosh.wav")
for tc in [6.9, 14.1, 21.2, 26.4, 28.8, 31.1, 33.1, 36.5, 40.9]: put(fx, fade(w, 0.2, 0.3), tc - pk(w), -30)
gg = load(FX + "/ia-teoli-geiger-4/teoli-geiger-counter-sounds-4.mp3", 1.0, 4.5); put(fx, fade(gg, 0.6, 1.2), 9.0, -30)
sw = load(FX + "/mixkit-2522-sci-fi-sweep/mixkit-2522-sci-fi-sweep.wav"); put(fx, fade(sw, 0.05, 0.3), 33.6, -28)
hit = lp(load(FX + "/mixkit-2913-epic-futuristic-movie-accent/mixkit-2913-epic-futuristic-movie-accent.wav"), 5000)
put(fx, fade(hit, 0.002, 1.5), 22.55 - pk(hit), -26); put(fx, fade(hit, 0.002, 2.5), 40.95 - pk(hit), -22)
mix = vo + mu * db(-8) + fx
tmp = OUT + ".pre.wav"
with wave.open(tmp, "wb") as f: f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
r = subprocess.run(["ffmpeg", "-v", "info", "-i", tmp, "-af", "loudnorm=I=-16:TP=-1:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(r[r.rindex("{"): r.rindex("}") + 1])
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", f"loudnorm=I=-16:TP=-1:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,aresample=48000", "-ar", "48000", OUT], check=True)
print("ok", j["input_i"])
