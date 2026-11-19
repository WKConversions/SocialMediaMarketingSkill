# Reads how things move in a reference: every unique frame of each motion event, side by side, with the
# moving region outlined and a speed curve, so an animation, transition or morph can be followed frame by
# frame (references/reading-references.md). A sheet a second apart shows *that* something changed; this
# shows *how*.
#   python3 motion_read.py ref.mp4 out_dir                     finds the events itself (onset to settle)
#   python3 motion_read.py ref.mp4 out_dir --at 4.2,9.8        or around given times (s)
#   [--max 16] [--w 300] [--cols 8] [--pad 3]
#
# Steps: (1) the cadence: an encode at 48 or 60 fps often repeats each drawn frame; duplicates are dropped,
# so every tile is a new drawing and the speed curve is per drawn frame. (2) The events: runs of change
# from onset to settle (not single peaks), merged across gaps under 3 drawn frames, padded by --pad.
# (3) One PNG per event: every drawn frame, timecode and frame number, the changed region outlined
# (red), and under the tiles the change per frame (bars) – read the curve: a tall start that decays is an
# ease-out (ARRIVE), a hump in the middle is ease-in-out, a flat run is linear, a spike is a cut,
# a second smaller hump after the main one is an overshoot or follow-through.
# Writes out_dir/event_NN.png, out_dir/events.txt (time, length, peak, curve shape). Needs opencv, numpy.
import argparse, os, cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("out")
ap.add_argument("--at"); ap.add_argument("--max", type=int, default=16)
ap.add_argument("--w", type=int, default=300); ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--pad", type=int, default=3)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30
frames, small = [], []
while True:
    ok, im = cap.read()
    if not ok: break
    frames.append(im)
    small.append(cv2.cvtColor(cv2.resize(im, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32))
raw = np.array([0.0] + [float(np.abs(small[i] - small[i - 1]).mean()) for i in range(1, len(small))])

# (1) cadence: keep frames that differ from the last kept one
keep = [0]
for i in range(1, len(small)):
    if np.abs(small[i] - small[keep[-1]]).mean() > 0.15: keep.append(i)
cad = len(small) / max(1, len(keep))
drawn_fps = fps / cad
d = np.array([0.0] + [float(np.abs(small[keep[j]] - small[keep[j - 1]]).mean()) for j in range(1, len(keep))])

# (2) events on drawn frames
if a.at:
    centres = [min(range(len(keep)), key=lambda j: abs(keep[j] / fps - float(t))) for t in a.at.split(",")]
    events = []
    for c in centres:
        s = e = c
        while s > 0 and d[s] > 0.25 and c - s < 30: s -= 1
        while e < len(d) - 1 and d[e + 1] > 0.25 and e - c < 30: e += 1
        events.append((s, e))
else:
    thr = max(0.25, float(np.percentile(d, 60)))
    events = []
    for j in np.flatnonzero(d > thr):
        if events and j - events[-1][1] <= 3: events[-1][1] = int(j)
        else: events.append([int(j), int(j)])
    events = [(s, e) for s, e in events]
    events = [ev for ev in events if ev[1] - ev[0] >= 2]
    events.sort(key=lambda ev: -float(d[ev[0]:ev[1] + 1].sum()))
    events = sorted(events[:a.max])

def shape(c):
    c = np.asarray(c, float)
    # a cut inside a move: one frame far above both neighbours' level splits the move into in and out
    if len(c) >= 5:
        pk = int(np.argmax(c)); nb = np.median(np.r_[c[max(0, pk - 3):pk], c[pk + 1:pk + 4]])
        if c[pk] > 3 * max(nb, 0.3) and 0 < pk < len(c) - 1:
            return (f"cut at drawn frame {pk}: before [{shape1(c[:pk])}] | after [{shape1(c[pk + 1:])}]"
                    f" (cut on the fastest frame if 'before' rises into it)")
    return shape1(c)

def rates(c):
    # ramp: how much the change grows per drawn frame over the (up to) 5 frames into the peak, excluding a cut frame;
    # settle: drawn frames after the peak until the change falls under a quarter of the pre-cut peak.
    c = np.asarray(c, float); pk = int(np.argmax(c))
    if pk < 3: return ""
    nb = np.median(np.r_[c[max(0, pk - 3):pk], c[pk + 1:pk + 4]]); cut = c[pk] > 3 * max(nb, 0.3)
    top = pk - 1 if cut else pk; seg = c[max(0, top - 4):top + 1]
    seg = seg[seg > 0.05]
    if len(seg) < 3: return ""
    ramp = float(np.exp(np.polyfit(np.arange(len(seg)), np.log(seg), 1)[0]))
    ref = c[top]; after = c[pk + 1:]; settle = next((i + 1 for i, v in enumerate(after) if v < ref / 4), len(after))
    return f"; ramp ×{ramp:.2f}/frame into the peak, settles in {settle} drawn frames"

def shape1(c):
    c = np.asarray(c, float)
    if len(c) < 3 or c.max() <= 0: return "too short"
    c = c / c.max(); pk = int(np.argmax(c)); n = len(c); rel = pk / (n - 1)
    if c.max() > 0 and (np.sort(c)[-2] if n > 1 else 0) < 0.35: return "spike (cut or flash)"
    tail = c[pk + 1:]
    second = tail.size > 3 and any(tail[k] > tail[k - 1] * 1.25 and tail[k] > 0.2 for k in range(2, tail.size))
    if c.std() < 0.18: base = "flat (linear or constant drift)"
    elif rel <= 0.25: base = "fast start, long decay (ease-out / ARRIVE)"
    elif rel >= 0.75: base = "slow start, sharp end (ease-in / exit)"
    else: base = "hump (ease-in-out)"
    return base + (", then a second hump (overshoot / follow-through / next layer)" if second else "")

log = [f"{a.video}: {fps:.3f} fps encoded, cadence {cad:.2f} (drawn at about {drawn_fps:.1f} fps), {len(keep)} drawn frames"]
for k, (s, e) in enumerate(events, 1):
    s2, e2 = max(0, s - a.pad), min(len(keep) - 1, e + a.pad)
    idx = list(range(s2, e2 + 1)); step = 1
    if len(idx) > 32:   # a long drift: every n-th drawn frame (written on the tiles), read the busy part again with --at
        step = -(-len(idx) // 32); idx = idx[::step]
    w = a.w; h = round(w * frames[0].shape[0] / frames[0].shape[1])
    tiles = []
    for j in idx:
        f = keep[j]; im = cv2.resize(frames[f], (w, h), interpolation=cv2.INTER_AREA)
        if j > 0:
            diff = cv2.resize(np.abs(small[f] - small[keep[j - step]]), (w, h)) > 8
            m = cv2.dilate(diff.astype(np.uint8), np.ones((5, 5), np.uint8))
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cs, -1, (40, 40, 235), 1)
        cv2.rectangle(im, (0, 0), (w, 16), (255, 255, 255), -1)
        cv2.putText(im, f"{f / fps:6.3f}s f{f} d{j - s2}" + (f" (every {step})" if step > 1 else ""),(3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % a.cols: tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[r:r + a.cols]) for r in range(0, len(tiles), a.cols)]
    curve = d[idx[0]:idx[-1] + 1]; bh = 70; W = rows[0].shape[1]
    bars = np.full((bh + 14, W, 3), 255, np.uint8); mx = max(1e-6, curve.max())
    for i, v in enumerate(curve):
        x = i * W // len(curve); x2 = (i + 1) * W // len(curve) - 2; y = bh - int(v / mx * (bh - 4))
        cv2.rectangle(bars, (x, y), (x2, bh), (200, 120, 40), -1)
    cv2.putText(bars, "change per drawn frame", (4, bh + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a.out, f"event_{k:02d}.png"), np.vstack(rows + [bars]))
    core = d[s:e + 1]
    log.append(f"event {k:02d}: {keep[s] / fps:6.2f}–{keep[e] / fps:6.2f} s, {e - s + 1} drawn frames "
               f"({(keep[e] - keep[s]) / fps:.2f} s), peak {core.max():.1f}, curve: {shape(core)}{rates(core)}")
open(os.path.join(a.out, "events.txt"), "w").write("\n".join(log) + "\n")
print("\n".join(log))
