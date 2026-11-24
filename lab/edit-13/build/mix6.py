"""Mix for the K.B explainer: the phone's voice, minimal UI-style effects only where something is visualised (found
sounds: Mixkit free licence, Kenney CC0; none synthesised here), optional background music under the voice.
    python -I mix6.py OUT.wav [music.mp3 start_s]   Master -14 LUFS, TP -1 dBTP."""
import json, subprocess, sys, wave
import numpy as np
from scipy.signal import butter, sosfilt
SR, DUR = 48000, 26.0; N = int(SR * DUR); OUT = sys.argv[1]
E = "/home/user/SocialMediaMarketingSkill/lab/edit-13"; FX = "/home/user/SocialMediaMarketingSkill/lab/marsops-03/sfx"
def load(p, s=0.0, d=None):
    c = ["ffmpeg", "-v", "error", "-ss", str(s)] + (["-t", str(d)] if d else []) + ["-i", p, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(c, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()
db = lambda g: 10 ** (g / 20)
lp = lambda x, hz: sosfilt(butter(4, hz, "low", fs=SR, output="sos"), x, axis=0)
def fade(x, a=0.005, b=0.05):
    n = len(x); e = np.ones(n, np.float32); k = min(n, int(a * SR)); e[:k] = np.linspace(0, 1, k); k = min(n, int(b * SR)); e[n - k:] *= np.linspace(1, 0, k); return x * e[:, None]
def put(bus, x, t, g):
    i = int(round(t * SR)); x = x[-i:] if i < 0 else x; i = max(i, 0); n = min(len(x), N - i)
    if n > 0: bus[i:i + n] += x[:n] * db(g)
pk = lambda x: int(np.argmax(np.abs(x).max(1))) / SR
def lufs(x):
    tmp = OUT + ".m.wav"
    with wave.open(tmp, "wb") as f: f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", tmp, "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(r[r.rindex("I:"):].split()[1])
vo = np.zeros((N, 2), np.float32); fx = np.zeros((N, 2), np.float32); mu = np.zeros((N, 2), np.float32)
v = load(f"{E}/build/voice.wav"); vo[:len(v)] += v; put(vo, load(f"{E}/build/outro.wav"), 20.3, 0); vo *= db(-16 - lufs(vo))                       # voice to -16 LUFS
tick = load(f"{FX}/kenney-interface-sounds-tick-002/tick_002.ogg"); sel = lp(load(f"{FX}/mixkit-3124-modern-technology-select/mixkit-3124-modern-technology-select.wav"), 8000)
clk = load(f"{FX}/mixkit-2577-interface-device-click/mixkit-2577-interface-device-click.wav"); tone = lp(load(f"{FX}/mixkit-2568-cool-interface-click-tone/mixkit-2568-cool-interface-click-tone.wav"), 9000)
air = load(f"{FX}/mixkit-168-fast-air-sweep-transition/mixkit-168-fast-air-sweep-transition.wav")
for t in (0.5, 2.62, 11.85, 20.6): put(fx, fade(sel), t - pk(sel), -27)
for k in range(5): put(fx, tick, 1.9 + k * 0.07, -31)                                              # the agencies turn into question marks                                      # a card appears
for t in (4.74, 7.1, 13.5, 23.23): put(fx, fade(air, 0.02, 0.2), t - pk(air) + 0.12, -30)       # the backdrop / panel moves in
for t in (6.3, 12.05, 14.6, 16.05, 17.8, 22.55): put(fx, fade(clk), t - pk(clk), -24)                           # the cursor clicks
for k in range(6): put(fx, tick, 8.3 + k * 0.17, -30)                                             # the timer running up
put(fx, fade(tone), 9.36 - pk(tone), -26); put(fx, fade(tone), 19.0 - pk(tone), -23)              # the timer lands; the result
if len(sys.argv) > 2:
    m = load(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0, DUR)[:N]; mu[:len(m)] += m
    mu *= db(-18.7 - lufs(mu))                                                                     # the approved level: voice - 2.7 LU
    e = np.abs(vo).max(1); w = int(0.3 * SR); env = np.convolve((np.convolve(e, np.ones(w) / w, "same") > 0.01).astype(float), np.ones(int(0.5 * SR)) / int(0.5 * SR), "same")
    t = np.arange(N) / SR; mu *= (db(-5.5 * env) * np.clip((DUR - t) / 1.2, 0, 1))[:, None]         # under the voice, up in the gaps and on the end card
mix = vo + fx + mu; tmp = OUT + ".pre.wav"
with wave.open(tmp, "wb") as f: f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
r = subprocess.run(["ffmpeg", "-v", "info", "-i", tmp, "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(r[r.rindex("{"): r.rindex("}") + 1])
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", f"loudnorm=I=-14:TP=-1:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,aresample=48000", "-ar", "48000", OUT], check=True)
print("pre", j["input_i"], "music" if len(sys.argv) > 2 else "no music")
