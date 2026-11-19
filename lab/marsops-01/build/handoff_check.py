# Finds pops: a place in the frame that changes in one frame when it should change smoothly.
#   python3 handoff_check.py film.mp4 [--cuts 120,452] [--strips out_dir]
#
# Typical pops: an object handed from one layer to another and redrawn at a different size (labels that
# jump when a card leaves a wall), a mask that stops clipping, a layout swapped for another, a layer that
# appears without a move. Dense optical flow predicts each frame from the last one, so everything that
# moves (wherever it came from) is explained; what is left is change no movement explains. Per tile,
# smooth work leaves similar amounts frame after frame; a pop spikes for one frame. Designed hard cuts (pass their
# frames with --cuts) are skipped. With --strips, each pop gets a strip of frames around it, cropped to
# the place it happened, to look at.  Exit code 1 when a pop is found. Needs: opencv-python-headless, numpy.
import cv2, numpy as np, os, sys

COLS, ROWS, SCALE = 8, 5, 0.25
RATIO = 4.0      # a spike is this many times its neighbours' change
FLOOR = 3.0      # and at least this much change (mean grey levels of 255) the motion doesn't explain

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

def main():
    path = sys.argv[1]
    cuts = {int(c) for c in arg("--cuts", "").split(",") if c}
    strips = arg("--strips")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    prev, R = None, []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            # dense optical flow from this frame back to the last one: everything that moved, wherever it
            # came from, is predicted; what the prediction misses is change no movement explains
            flow = cv2.calcOpticalFlowFarneback(g, prev, None, 0.5, 3, 15, 3, 5, 1.1, 0)
            H, W = g.shape
            gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            pred = cv2.remap(prev, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            res = cv2.blur(np.abs(g.astype(np.float32) - pred.astype(np.float32)), (3, 3))
            th, tw_ = H // ROWS, W // COLS
            R.append([float(res[r*th+2:(r+1)*th-2, c*tw_+2:(c+1)*tw_-2].mean()) for r in range(ROWS) for c in range(COLS)])
        prev = g
    R = np.array(R); n, T = R.shape          # R[t] = change from frame t to t+1
    hits = np.zeros((n, T), bool); score = np.zeros((n, T))
    for t in range(5, n - 5):   # the first and last frames have no neighbours to compare with
        base = np.median(np.concatenate([R[t - 3:t], R[t + 1:t + 4]]), axis=0)
        hits[t] = (R[t] > FLOOR) & (R[t] > RATIO * (base + 0.4)); score[t] = R[t] / (base + 0.4)
    # something fast travelling across the frame lights neighbouring tiles in neighbouring frames; a pop
    # lights its place in one frame only. Drop every hit that is part of such a chain.
    def neighbours(k):
        r, c = divmod(k, COLS)
        return [rr * COLS + cc for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2) if 0 <= rr < ROWS and 0 <= cc < COLS]
    keep = hits.copy()
    for t in range(n):
        for k in np.where(hits[t])[0]:
            near = neighbours(k)
            if any(hits[t2, near].any() for t2 in (t - 2, t - 1, t + 1, t + 2) if 0 <= t2 < n):
                keep[t, k] = False
    events = []
    for t in range(n):
        if t + 1 in cuts or t in cuts: continue
        if hits[t].mean() > 0.6:
            events.append((t + 1, "cut", np.where(hits[t])[0], float(score[t].max())))
        elif keep[t].any():
            events.append((t + 1, "pop", np.where(keep[t])[0], float(score[t][keep[t]].max())))
    pops = [e for e in events if e[1] == "pop"]
    for f, kind, tiles, score in events:
        cells = ", ".join(f"r{t // COLS}c{t % COLS}" for t in tiles[:8])
        print(f"{'POP' if kind == 'pop' else 'cut'} at {f / fps:.2f} s (frame {f}): {score:.1f}× the change around it, tiles {cells}{' …' if len(tiles) > 8 else ''}")
    if pops:
        top = sorted(pops, key=lambda e: -e[3])[:5]
        print(f"{len(pops)} pops; the strongest at frames {', '.join(str(e[0]) for e in top)}. Look at each strip: a designed")
        print("text change can show here; a blink, a jump in size or a layer appearing without a move is a bug.")
    if not events:
        print(f"no pops in {n + 1} frames")
    elif not pops:
        print("no pops; the cuts above change the whole frame at once: fine if they are designed (pass them with --cuts)")
    if strips and pops:
        os.makedirs(strips, exist_ok=True)
        cap2 = cv2.VideoCapture(path)
        for f, _, tiles, _ in pops:
            rs = [t // COLS for t in tiles]; cs = [t % COLS for t in tiles]
            H, W = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            th, tw_ = H // ROWS, W // COLS
            y0, y1 = max(0, (min(rs) - 1) * th), min(H, (max(rs) + 2) * th); x0, x1 = max(0, (min(cs) - 1) * tw_), min(W, (max(cs) + 2) * tw_)
            tiles_img = []
            for k in range(f - 3, f + 3):
                cap2.set(cv2.CAP_PROP_POS_FRAMES, k); _, fr = cap2.read()
                crop = cv2.resize(fr[y0:y1, x0:x1], (480, int(480 * (y1 - y0) / (x1 - x0))))
                cv2.putText(crop, f"{k}{'  <- pop' if k == f else ''}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                tiles_img.append(crop)
            cv2.imwrite(os.path.join(strips, f"pop_{f:05d}.png"), np.hstack(tiles_img))
        print(f"strips: {strips}/pop_*.png (frames {', '.join(str(e[0]) for e in pops)})")
    sys.exit(1 if pops else 0)

main()
