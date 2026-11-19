# Measures how a film hands motion from one move to the next (vault: Principles/Hand the motion on).
#   python3 relay_check.py film.mp4 [more.mp4 ...]
# Finds every motion event on drawn frames (duplicates from cadence removed), then the gap from one event's settle
# to the next event's start. Reports the share of gaps that are relays (1–6 drawn frames: the next move picks up
# as the last one lands), overlaps (0: one blob of motion) and pauses (over 12 drawn frames), and the median gap.
import sys, cv2, numpy as np
for p in sys.argv[1:]:
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 30; sm = []
    while True:
        ok, im = cap.read()
        if not ok: break
        sm.append(cv2.cvtColor(cv2.resize(im, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32))
    keep = [0]
    for i in range(1, len(sm)):
        if np.abs(sm[i] - sm[keep[-1]]).mean() > 0.15: keep.append(i)
    d = np.array([0.0] + [float(np.abs(sm[keep[j]] - sm[keep[j - 1]]).mean()) for j in range(1, len(keep))])
    if len(d) > 4 * fps: d = d[: len(d) - int(2 * fps * len(keep) / len(sm))]   # end card excluded
    thr = max(0.4, float(np.percentile(d, 50)))
    ev = []
    for j in np.flatnonzero(d > thr):
        if ev and j - ev[-1][1] <= 1: ev[-1][1] = int(j)
        else: ev.append([int(j), int(j)])
    ev = [e for e in ev if e[1] - e[0] >= 1]
    gaps = np.array([b[0] - a[1] - 1 for a, b in zip(ev, ev[1:])])
    if len(gaps) == 0: print(f"{p}: too few events"); continue
    relay = ((gaps >= 1) & (gaps <= 6)).mean(); pause = (gaps > 12).mean()
    name = p.split("/")[-2] if p.endswith("v.mp4") else p.split("/")[-1]
    print(f"{name:>16}: {len(ev)} events, gaps median {np.median(gaps):.0f} drawn frames; relays (1–6) {100*relay:.0f}%, "
          f"pauses (>12) {100*pause:.0f}%")
