# Finds the busy look: too many separate things on screen at once, each too small to read. Karl, on K.B (approved,
# but): "a bit too busy so you can't see what's happening". Motion wasn't the cause (K.B moves less than most of
# his references); the count of things was.
#   python3 busy_check.py film.mp4 [--profile] [--json out.json] [--ui]
# Four times a second it finds the background (the most common colour), joins letters into words and words into
# lines, and counts the separate shapes left: a headline is one thing, a card is one thing, a row of five icons is
# five. It also measures how much of the frame they cover.
# Calibrated on Karl's 25 reference films against the three K.B renders:
#                          things on screen (median)   time with 8 or more things   frame covered (median)
#   references (median)    4   (range 3–7)             25%  (at most 46%)            26%  (range 3–56%)
#   K.B v1 / v2 / v3       11 / 9 / 9                  87% / 70% / 64%               7% / 7% / 11%
# Busy when the film's median is above 6, or 8+ things fill more than 35% of the time (45% with --ui, for films
# that are mostly an interface). The crowded stretches are listed so you know where to take things away.
# Exit code 1 when busy. Needs opencv-python-headless and numpy.
import argparse, json, sys
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--profile", action="store_true"); ap.add_argument("--json")
ap.add_argument("--ui", action="store_true", help="the film is mostly an interface: allow more crowded time")
a = ap.parse_args()

cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30; step = max(1, int(round(fps / 4)))
rows, i = [], 0
while True:
    ok, fr = cap.read()
    if not ok: break
    if i % step: i += 1; continue
    im = cv2.resize(fr, (480, 270), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.int16)
    q = (lab // 12).reshape(-1, 3); keys, cnt = np.unique(q, axis=0, return_counts=True); bg = keys[cnt.argmax()] * 12 + 6
    fg = (np.abs(lab - bg).max(axis=2) > 22).astype(np.uint8)
    k = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))       # letters → words → lines
    n, _, st, _ = cv2.connectedComponentsWithStats(k, 8)
    things = int((st[1:, cv2.CC_STAT_AREA] > 60).sum())
    rows.append((i / fps, things, float(fg.mean())))
    i += 1

r = np.array(rows); t, th, cov = r[:, 0], r[:, 1], r[:, 2]
sm = np.convolve(th, np.ones(3) / 3, mode="same")
med, share, cover = float(np.median(th)), float(np.mean(sm >= 8)), float(np.median(cov))
limit = 0.45 if a.ui else 0.35
busy = med > 6 or share > limit
runs, j = [], 0
while j < len(sm):
    if sm[j] >= 8:
        k = j
        while k + 1 < len(sm) and sm[k + 1] >= 8: k += 1
        if (k - j + 1) / 4 >= 1.0: runs.append((t[j], t[k], int(th[j:k + 1].max())))
        j = k + 1
    else: j += 1
for s0, s1, mx in runs:
    print(f"CROWDED {s0:.2f}–{s1 + 0.25:.2f} s: up to {mx} separate things on screen")
print(f"things on screen: median {med:.0f} (references 3–7, median 4); 8 or more for {share * 100:.0f}% of the time "
      f"(references median 25%, limit {limit * 100:.0f}%); frame covered {cover * 100:.0f}% (references median 26%)"
      + ("  → BUSY: fewer, bigger things" if busy else ""))
if a.profile:
    for s in range(0, len(th), 4):
        print(f"{t[s]:6.1f}s  things {' '.join(f'{int(x):2d}' for x in th[s:s + 4])}  covered {cov[s:s + 4].mean() * 100:4.0f}%")
if a.json:
    json.dump({"t": t.tolist(), "things": th.tolist(), "cover": cov.tolist(), "median": med, "share8": share, "busy": busy,
               "crowded": runs}, open(a.json, "w"))
sys.exit(1 if busy else 0)
