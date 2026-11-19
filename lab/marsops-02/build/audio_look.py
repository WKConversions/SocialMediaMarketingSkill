# Shows a sound file as a picture: a log-frequency spectrogram over a loudness curve, with second marks
# and optional marks at given times.  python3 audio_look.py file.wav out.png [--marks 8.89,15.56] [--title t]
# For judging music and mixes by eye: sections arriving where planned, the low end, harshness, clipping,
# the voice sitting above the bed.  Needs ffmpeg, numpy, Pillow.
import subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
SR = 22050
path, out = sys.argv[1], sys.argv[2]
marks = [float(m) for m in sys.argv[sys.argv.index("--marks") + 1].split(",")] if "--marks" in sys.argv else []
title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else path
# decoded as stereo and averaged here: ffmpeg's own mono downmix sums the channels and overstates the peak
st = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, dtype=np.float32).reshape(-1, 2)
x = st.mean(axis=1)
N, H = 2048, 256; W = 1800
frames = [x[i:i + N] * np.hanning(N) for i in range(0, len(x) - N, H)]
S = 20 * np.log10(np.abs(np.fft.rfft(np.array(frames), axis=1)) + 1e-6)
freqs = np.fft.rfftfreq(N, 1 / SR)
rows = 260; logf = np.geomspace(40, 11000, rows)
img = np.stack([S[:, np.argmin(np.abs(freqs - f))] for f in logf[::-1]])
img = np.clip((img - (img.max() - 80)) / 80, 0, 1)
col = (np.stack([img * 255, img ** 0.7 * 210, (1 - img) * 90 + img * 60], -1)).astype(np.uint8)
spec = Image.fromarray(col).resize((W, rows * 2))
rms = np.sqrt(np.array([np.mean(f ** 2) for f in frames]) + 1e-12); db = 20 * np.log10(rms / rms.max())
canvas = Image.new("RGB", (W + 60, rows * 2 + 200), "white"); canvas.paste(spec, (60, 30)); d = ImageDraw.Draw(canvas)
d.text((60, 8), title + f"   peak {20*np.log10(np.abs(st).max()+1e-9):.1f} dBFS", fill=(0, 0, 0))
for f in (100, 1000, 10000):
    y = 30 + int(np.argmin(np.abs(logf[::-1] - f)) * 2); d.text((4, y - 6), f"{f if f < 1000 else str(f // 1000) + 'k'}Hz", fill=(0, 0, 0))
y0 = rows * 2 + 40; dur = len(x) / SR
pts = [(60 + i * W / len(db), y0 + 120 - max(0, (v + 40) / 40 * 120)) for i, v in enumerate(db)]
d.line(pts, fill=(30, 40, 90), width=2)
for s in range(int(dur) + 1):
    X = 60 + s / dur * W; d.line([X, y0 + 120, X, y0 + 128], fill=(0, 0, 0)); d.text((X + 2, y0 + 130), str(s), fill=(90, 90, 90))
for m in marks:
    X = 60 + m / dur * W; d.line([X, 30, X, y0 + 120], fill=(220, 40, 40), width=2)
canvas.save(out)
