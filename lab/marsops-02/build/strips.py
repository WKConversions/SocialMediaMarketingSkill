# Transition strips: frames around each moment of change, side by side, to look at before the final render.
#   python3 strips.py film.mp4 out_dir                 # finds the transitions itself (the biggest changes)
#   python3 strips.py film.mp4 out_dir --at 140,146    # or around given frames
#   [--every 2] [--before 6] [--after 8] [--max 24]
#
# Problems that are invisible at full speed sit in transitions: two captions overlapping for six frames,
# an object that grows as it is handed over, a layer that blinks. Each strip shows frames from 6 before to
# 8 after the moment, every 2 frames, with frame numbers; one PNG per transition plus contact sheets of
# all of them. Needs: opencv-python-headless, numpy.
import cv2, numpy as np, os, sys

def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

path, out = sys.argv[1], sys.argv[2]
every, before, after, most = int(arg("--every", 2)), int(arg("--before", 6)), int(arg("--after", 8)), int(arg("--max", 24))
os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
at = arg("--at", "")
if at:
    moments = [int(x) for x in at.split(",")]
else:
    # the change between frames, small grey, then the peaks of a smoothed curve, at least 20 frames apart
    prev, d = None, []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        d.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    d = np.convolve(np.array(d), np.ones(5) / 5, mode="same")
    order = np.argsort(-d)
    moments = []
    for i in order:
        if d[i] < 0.6 or len(moments) >= most: break
        if all(abs(i - m) >= 20 for m in moments): moments.append(int(i))
    moments.sort()
W = 360
strips = []
for m in moments:
    row = []
    for k in range(m - before, m + after + 1, every):
        k = min(max(k, 0), n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
        if not ok: continue
        im = cv2.resize(fr, (W, W * 9 // 16), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (W - 1, 26), (255, 255, 255), -1)
        cv2.putText(im, f"{k}  {k / fps:.2f}s", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200) if k == m else (40, 40, 40), 1, cv2.LINE_AA)
        row.append(im)
    strip = np.hstack(row)
    cv2.imwrite(os.path.join(out, f"strip_{m:05d}.png"), strip)
    strips.append(strip)
# contact sheets, six strips each, for looking at several at once
for s in range(0, len(strips), 6):
    sheet = np.vstack(strips[s:s + 6])
    cv2.imwrite(os.path.join(out, f"sheet_{s // 6 + 1:02d}.png"), cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA))
print(f"{len(strips)} strips in {out}: frames {', '.join(map(str, moments))}")
