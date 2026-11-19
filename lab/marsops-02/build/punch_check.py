# How often a film makes a big visual change (a scene turn: a cut, a push-through, a full-frame wipe), against references.
#   python3 punch_check.py film.mp4 [more.mp4 ...] [--thr 25]
# Per film: big moments (drawn frames whose mean change is over --thr, on 0–255 grey, merged within 0.3 s), the
# median and longest gap between them, and a timeline. References measured 2026-10: median gap 1.5–3.5 s.
# A film with only local changes on one held stage reads as a page being edited, not a film moving on.
import sys, cv2, numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]; thr = 25.0
if "--thr" in sys.argv: thr = float(sys.argv[sys.argv.index("--thr") + 1]); args.remove(str(sys.argv[sys.argv.index("--thr") + 1]))
for p in args:
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 30; prev = None; d = []; i = 0
    while True:
        ok, im = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(im, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        d.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g
    d = np.array(d); dur = len(d) / fps; body = d[: max(1, len(d) - int(2 * fps))] if dur > 6 else d
    t = [i / fps for i in np.flatnonzero(body > thr)]; m = []
    for x in t:
        if not m or x - m[-1] > 0.3: m.append(x)
    gaps = np.diff([0.0] + m + [len(body) / fps])
    line = "".join("|" if any(int(x) == s for x in m) else "." for s in range(int(len(body) / fps) + 1))
    print(f"{p.split('/')[-2] if p.endswith('v.mp4') else p.split('/')[-1]:>14}: {len(m)} big moments in {len(body)/fps:.0f} s, "
          f"median gap {np.median(gaps):.1f} s, longest {gaps.max():.1f} s  {line}")
