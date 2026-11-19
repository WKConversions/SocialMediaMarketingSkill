# Measures how constantly and how much a video moves: python3 motion_check.py out.mp4 [--profile]
# Needs numpy and opencv-python-headless. Ignores the last 2 seconds (the end card) on films longer than 4 s.
# --profile adds one line per second: the share of the frame in motion and any still frames, to find
# dead, flat or over-busy stretches (the rhythm pass in evaluation/quality-check.md).
import cv2, numpy as np, sys, collections
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS); prev = None; frac = []
fine = collections.deque(maxlen=int(2.2 * (fps or 30)) + 8)   # the last ~2 s at 320x180, for the end card
while True:
    ok, fr = cap.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(fr, (160, 90), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine.append(cv2.cvtColor(cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32))
    if prev is not None: frac.append(float((np.abs(g - prev) > 6).mean()))   # share of the frame that changed
    prev = g
frac = np.array(frac); fps0 = fps or 30
# cadence: an encode at 48 or 60 fps of a film drawn at 24 or 30 (or animated on twos) repeats frames, not always
# in an even pattern; a held frame next to a drawn one is not a still moment. A frame is still only when it AND
# its neighbours are still (no drawing within one frame either side), so holds of up to 2 encoded frames don't count.
dup = float((frac < 0.001).mean())
if dup > 0.3:
    near = np.maximum.reduce([frac, np.r_[frac[1:], 0], np.r_[0, frac[:-1]]])
    print(f"cadence: {100 * dup:.0f}% of encoded frames repeat the previous one (held drawings); "
          f"still = no new drawing within a frame either side")
    frac = near
cut = int(2 * fps) if len(frac) > 4 * fps else 0
body = frac[:len(frac) - cut]; still = body < 0.001; runs, r = [0], 0
for s in still: r = r + 1 if s else 0; runs.append(r)
print(f"moving in {100 * (1 - still.mean()):.0f}% of frames (target 90+), "
      f"median {100 * np.median(frac):.1f}% of the frame in motion (information, not a target: K.B v3, approved, 1.7; references 0.5–6.6), "
      f"longest still {max(runs) / fps:.2f} s (target 0.8 or less)")
if cut:   # the end card, judged on its own (vault: Principles/End cards keep moving)
    # slow end-card moves (tracking, drift, a glow) are under a pixel per frame: compare each frame with the one
    # 0.2 s earlier at 320x180, so slow motion adds up; a frame is still when nothing changed over that 0.2 s.
    fr_ = list(fine); lag = max(1, round(0.2 * fps0)); n_end = min(len(fr_) - lag, int(2 * fps0))
    ch = [float((np.abs(fr_[i] - fr_[i - lag]) > 4).mean()) for i in range(len(fr_) - n_end, len(fr_))]
    st = np.array(ch) < 0.0005; er = [0]; r = 0
    for x in st: r = r + 1 if x else 0; er.append(r)
    print(f"end card (last 2 s): moving in {100 * (1 - st.mean()):.0f}% of frames (over 0.2 s windows), longest still "
          f"{max(er) / fps0:.2f} s (target 0.5 s or less: the lockup holds its place, something keeps moving to the end)")
if '--profile' in sys.argv:
    n = max(1, round(fps))
    for i in range(0, len(frac), n):
        seg = frac[i:i + n]; m = 100 * float(seg.mean()); k = int((seg < 0.001).sum())
        print(f"{i / fps:6.1f}s {m:5.1f}% {'#' * min(60, round(m * 4)):<60}" + (f" still {k}" if k else ""))
