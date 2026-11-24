"""Find the offset between the phone recording (the audio we keep) and the second angle, from their audio.
Cross-correlation of the band-passed waveforms at 16 kHz, checked in three separate windows (drift).
    python -I sync.py   -> src/sync.json  {"offset_s": t}: angle2 time = phone time + offset_s"""
import json, subprocess
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
E = "/home/user/SocialMediaMarketingSkill/lab/edit-13"; SR = 16000
def load(p):
    x = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout, np.float32)
    return sosfilt(butter(4, [200, 4000], "band", fs=SR, output="sos"), x)
a, b = load(f"{E}/src/phone.mov"), load(f"{E}/src/angle2.mov")
def xc(a, b):
    c = fftconvolve(b, a[::-1]); k = int(np.argmax(np.abs(c))); lag = k - (len(a) - 1)
    return lag / SR, float(np.abs(c[k]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
off, q = xc(a, b); print("whole: angle2 = phone + %.4f s, corr %.3f" % (off, q))
res = []
for t0 in (5, 22, 40):
    seg = a[int(t0 * SR): int((t0 + 10) * SR)]; i0 = int((t0 + off - 1.5) * SR); win = b[max(0, i0): i0 + int(13 * SR)]
    o, qq = xc(seg, win); o = o + max(0, i0) / SR - t0; res.append((t0, round(o, 4), round(qq, 3))); print("window at %2d s: %.4f s, corr %.3f" % (t0, o, qq))
json.dump({"offset_s": round(off, 4), "corr": round(q, 3), "windows": res, "drift_ms_over_35s": round((res[-1][1] - res[0][1]) * 1000, 1)}, open(f"{E}/src/sync.json", "w"), indent=1)
