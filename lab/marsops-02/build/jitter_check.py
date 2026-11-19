# Finds shake: things that move in steps instead of gliding.  python3 jitter_check.py film.mp4 [--png worst.png]
#
# The motion check measures how much moves; this measures how smoothly. The frame is cut into a grid of
# tiles and each tile's shift between frames is tracked with phase correlation (a reading counts only on
# a textured tile with a clear correlation peak). A shake has one signature: during slow, steady motion
# one frame's velocity sticks out from both neighbours, which agree with each other (+1.2, +1.2, +1.2,
# −2.5, +1.2 px …), again and again in the same place. Smooth motion, eased or not, never does that, and
# a fade or a word appearing can't repeat it three times in 20 frames.
#
# Typical causes, in the order they have happened: an element positioned with left/top under a zoomed
# camera (box offsets snap to whole pixels; position with transform instead); a camera that tracks a
# moving object which itself snaps; text re-laid out every frame. Exit code 1 when a shake (an error) is found;
# "look" lines are small irregular jumps worth a glance in a transition strip.
# Held drawings: a film animated on twos (or with frames repeated by an encode) shows 0, 2×, 0, 2× velocities,
# the very pattern of a shake. Frames identical to the last drawing are skipped and each velocity is divided by
# the frames that passed, so SHAKE means positions that jump, not drawings that hold. Holds in the middle of
# motion are reported apart: on a film drawn on ones (our renders) they are a STUTTER (a stalled frame); on a
# film with many held drawings (on twos) they are its style.
# Needs: opencv-python-headless, numpy.
import cv2, numpy as np, sys

COLS, ROWS, SCALE = 8, 5, 0.5   # analysis grid, and the analysis resolution (half of 1080p)
SLOW_PX = 10.0                  # only motion slower than this (px/frame, full resolution) is judged
SPIKE_PX = 1.0                  # how far one frame's velocity must stick out from its neighbours
MIN_SPIKES = 3                  # spikes within 20 frames, in one tile, that make a shake
WIN = 20
DETAIL = 4.0                    # high-pass std a tile needs to count as textured (edges, type, UI)

def detail(t):
    return float((t - cv2.GaussianBlur(t, (0, 0), 3)).std())

def main():
    path = sys.argv[1]
    png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    prev, V, OK = None, [], []
    idx, held, k, elapsed = [], [], -1, 1     # encoded index of each drawing; held frames
    while True:
        ok, fr = cap.read()
        if not ok: break
        k += 1
        g = cv2.cvtColor(cv2.resize(fr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None and float(np.abs(g - prev).mean()) < 0.15:
            held.append(k); elapsed += 1; continue
        H, W = g.shape; th, tw_ = H // ROWS, W // COLS
        if prev is None:
            last = [g[r*th:(r+1)*th, c*tw_:(c+1)*tw_].copy() for r in range(ROWS) for c in range(COLS)]; el = [1] * (ROWS * COLS)
        else:
            win = cv2.createHanningWindow((tw_, th), cv2.CV_32F)
            vs, oks = [], []
            for r in range(ROWS):
                for c in range(COLS):
                    t = r * COLS + c
                    # per tile, against its last drawing: one layer can hold on twos while another moves on ones
                    a = last[t]; b = g[r*th:(r+1)*th, c*tw_:(c+1)*tw_].copy()
                    if float(np.abs(b - a).mean()) < 0.25 and el[t] < 3:
                        el[t] += elapsed; vs.append((0.0, 0.0)); oks.append(False); continue
                    # copies: phaseCorrelate writes its window back into a slice it is given
                    (dx, dy), resp = cv2.phaseCorrelate(a.copy(), b.copy(), win)
                    # texture = fine detail, not brightness spread: a smooth gradient with a drifting glow and grain
                    # has a high std but nothing to track, and phase correlation then reads the grain as jumps
                    good = min(detail(a), detail(b)) > DETAIL and resp > 0.4 and abs(dx) < 12 and abs(dy) < 12
                    e = el[t] + elapsed - 1 if el[t] < 3 else elapsed
                    vs.append((dx / SCALE / e, dy / SCALE / e)); oks.append(good)
                    last[t] = b; el[t] = 1
            V.append(vs); OK.append(oks); idx.append(k)
        prev = g; elapsed = 1
    v, ok = np.array(V), np.array(OK)        # [frame, tile, 2], [frame, tile]
    n, T = ok.shape
    slow = ok & (np.abs(v).max(axis=2) < SLOW_PX)
    # a spike at t: both neighbours valid and close to each other, t far from both, in the same direction
    spike = np.zeros((n, T), bool); size = np.zeros((n, T))
    for t in range(1, n - 1):
        both = slow[t - 1] & slow[t] & slow[t + 1]
        d1, d2 = v[t] - v[t - 1], v[t] - v[t + 1]
        dev = np.minimum(np.abs(d1), np.abs(d2)).max(axis=1) * (np.sign(d1) == np.sign(d2)).all(axis=1)
        agree = np.abs(v[t + 1] - v[t - 1]).max(axis=1) < 0.5 * np.maximum(dev, 1e-6)
        spike[t] = both & agree & (dev > SPIKE_PX); size[t] = np.where(spike[t], dev, 0)
    # out-and-back: one frame displaced and back (a pop): v[t] and v[t+1] deviate from the steady speed by about
    # the same amount in opposite directions, while v[t-1] and v[t+2] agree
    for t in range(1, n - 2):
        ends = slow[t - 1] & slow[t] & slow[t + 1] & slow[t + 2]
        base = (v[t - 1] + v[t + 2]) / 2
        d, e = v[t] - base, v[t + 1] - base
        steady = np.abs(v[t + 2] - v[t - 1]).max(axis=1) < SPIKE_PX
        mag = np.minimum(np.abs(d), np.abs(e)).max(axis=1)
        back = (np.sign(d) == -np.sign(e)).all(axis=1)
        pop = ends & steady & back & (mag > SPIKE_PX) & (np.abs(d + e).max(axis=1) < 0.5 * mag)
        spike[t] |= pop; size[t] = np.maximum(size[t], np.where(pop, mag, 0))
    pad = np.pad(spike.astype(int), ((WIN // 2, WIN // 2), (0, 0)))
    count = np.stack([pad[i:i + WIN].sum(axis=0) for i in range(n)])
    bad = count >= MIN_SPIKES
    found, i = [], 0
    anyb = bad.any(axis=1)
    while i < n:
        if anyb[i]:
            j = i
            while j < n and anyb[j]: j += 1
            tiles = np.where(bad[i:j].any(axis=0))[0]
            a0, b0 = max(0, i - WIN // 2), min(n, j + WIN // 2)
            # regular steps (a constant slow speed snapping to the pixel grid) or many tiles at once (the
            # camera) make an error; a few irregular jumps in one place are a note to look at
            best = max(tiles, key=lambda t: spike[a0:b0, t].sum())
            at = np.where(spike[a0:b0, best])[0]
            periodic = len(at) >= 3 and np.std(np.diff(at)) <= 1.0
            peak = float(size[a0:b0].max())
            # frame-rate conversion: the whole frame (3+ tiles) double-steps at a fixed period, 4–6 drawings apart
            glob = np.where((spike[a0:b0].sum(axis=1)) >= 3)[0]
            gd = np.diff(glob) if len(glob) >= 3 else np.array([])
            conv = int(np.median(gd)) if len(gd) and 3 <= np.median(gd) <= 6 and (np.abs(gd - np.median(gd)) <= 1).mean() >= 0.7 else 0
            major = (len(tiles) >= 4 and peak >= 1.5) or (periodic and peak >= 1.0)
            found.append((idx[a0], idx[min(b0, n) - 1], tiles, peak, int(count[i:j].max()), major, periodic, conv))
            i = j
        else:
            i += 1
    majors = [f for f in found if f[5] or f[7]]     # periodic whole-frame steps (JUDDER) are defects too
    # held frames during motion: a drawing that holds while the drawings either side of it move
    total = k + 1; share = len(held) / max(1, total)
    moving = {idx[t]: float(np.median(np.abs(v[t][ok[t]]).max(axis=1))) if ok[t].any() else 0.0 for t in range(n)}
    stalls = [h for h in held if moving.get(h + 1, 0) > 0.5 and any(moving.get(h - j, 0) > 0.5 for j in (1, 2))]
    if share > 0.3:
        print(f"held drawings: {100 * share:.0f}% of frames repeat the previous one (animated on twos or a converted encode); judged per drawing")
    elif len(stalls) >= 4 and (np.abs(np.diff(stalls)[:20] - 5) <= 0).mean() >= 0.6:
        print(f"PULLDOWN: a frame repeats every 5 frames ({', '.join(map(str, stalls[:6]))} …): 24 fps material in a 30 fps file; "
              f"motion stutters 6 times a second. Render and deliver at one rate")
    elif stalls:
        print(f"STUTTER: {len(stalls)} frame(s) held in the middle of motion, at frames {', '.join(map(str, stalls[:12]))}{' …' if len(stalls) > 12 else ''}"
              f" (a stalled render frame or a duplicated frame in the encode: the motion stops for a frame and jumps)")
    if not found:
        print(f"no shake in {n + 1} frames (most spikes in one tile in {WIN} frames: {int(count.max())}; a shake needs {MIN_SPIKES})")
    for a, b, tiles, peak, cnt, major, periodic, conv in found:
        cells = ", ".join(f"r{t // COLS}c{t % COLS}" for t in tiles[:8])
        if conv:
            print(f"JUDDER {a / fps:.2f}–{b / fps:.2f} s (frames {a}–{b}): the whole frame double-steps every {conv} frames: a frame "
                  f"dropped by a frame-rate conversion (rendered at one rate, encoded at another: render at the delivery rate), or a "
                  f"camera that snaps to whole pixels on a fixed beat (position with sub-pixel transforms).")
            continue
        kind = "SHAKE" if major else "look "
        why = "regular steps" if periodic else ("many tiles" if len(tiles) >= 4 else "irregular, one place")
        print(f"{kind} {a / fps:.2f}–{b / fps:.2f} s (frames {a}–{b}): {cnt} jumps of up to {peak:.1f} px in {WIN} frames ({why}), tiles {cells}{' …' if len(tiles) > 8 else ''}")
    if not majors and found:
        print("no shake or judder; the 'look' lines are small irregular jumps: check them in a transition strip (a rolling number or a repeating pattern can cause them)")
    if png and found:
        a, b, tiles = max(found, key=lambda f: (f[5], len(f[2]) * f[4]))[:3]
        a, b = idx.index(a) if a in idx else 0, idx.index(b) if b in idx else 0
        a, b = idx[a], idx[b]
        cap2 = cv2.VideoCapture(path); cap2.set(cv2.CAP_PROP_POS_FRAMES, (a + b) // 2); _, fr = cap2.read()
        H, W = fr.shape[:2]; th, tw_ = H // ROWS, W // COLS
        for t in tiles:
            r, c = divmod(int(t), COLS)
            cv2.rectangle(fr, (c * tw_, r * th), ((c + 1) * tw_, (r + 1) * th), (0, 0, 255), 4)
        cv2.imwrite(png, fr)
    sys.exit(1 if majors else 0)

main()
