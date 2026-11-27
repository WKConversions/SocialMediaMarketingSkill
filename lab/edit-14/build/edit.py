"""10 s high-end flash edit (vertical 1080x1920, 30 fps) of the late-night coding clips.
- Shots cut on the 8th-note grid of the track ("Trap Electro Vibes", Alejandro Magana / Mixkit, free licence), not in
  file order: a story order (hook on the fan, the person at the desk, hands, code, silhouettes, back to the desk).
- Object-flash transitions (the reference's "object flash"): the outgoing shot's subject stays in colour while its
  background snaps to a hard black/white threshold and burns to white; the incoming shot's subject flickers as a white
  silhouette twice. Camera shutter clicks on every flash (Mixkit free licence). Plus exposure flashes and punch cuts.
- Masks: BiRefNet-lite (MIT), cleaned to one smooth solid shape; in shots of the person a body is added below the
  head/torso so the silhouette never breaks into pieces.
- Privacy: all screen text is blurred (bright, low-saturation screen areas; the text close-up is blurred whole).
- Light grade, digital camera moves (push, punch, shake), grain, vignette, bloom.
    python -I edit.py [--demo]      -> out/frames/*.jpg, out/audio.wav, out/<name>.mp4
"""
import json, math, os, subprocess, sys, wave
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg import seg

E = "/home/user/SocialMediaMarketingSkill/lab/edit-14"; FPS = 30; W, H = 1080, 1920; N = 300
MUSIC = f"{E}/music/trap-electro-vibes.mp3"; M0 = 11.98            # the drop's first kick (12.144 s) lands at 0.164 s
G = lambda k: 0.164 + 0.32515 * k                                  # 8th-note grid (184.6 BPM double time / 92.3 BPM)
PERSON = {"7402", "7403", "7405", "7406"}
TEXT_CLOSEUP = {"7401"}
# (start frame, clip, source in-point s, speed, transition INTO this shot). Designed by hand so nothing repeats: shot
# lengths from 3 to 38 frames in no fixed order, bursts of 1, 3, 5 and 2 quick cuts with different clips each time, and
# the transitions mixed: EF exposure flash; OF1/OF2/OF3 object flash with 1/2/3 white flickers; HC hard cut with a
# one-frame black/white blink and a click; BU burst cut with a click; HX a dry cut (no click) to breathe.
SHOTS = [(0, "7400", 0.15, 1.2, "EF"), (8, "7407", 0.10, 1.1, "HC"), (19, "7402", 0.60, 1.1, "OF2"),
         (35, "7410", 0.50, 1.0, "BU"), (38, "7404", 1.20, 1.0, "HX"),
         (45, "7408", 0.25, 1.0, "BU"), (48, "7401", 3.20, 1.0, "BU"), (51, "7400", 1.00, 1.0, "BU"),
         (55, "7406", 2.60, 1.2, "EF"), (77, "7403", 1.00, 1.1, "OF1"), (89, "7408", 0.90, 1.0, "BU"),
         (93, "7402", 3.10, 1.0, "EF"), (120, "7401", 6.20, 1.0, "BU"), (123, "7407", 0.55, 1.0, "HC"),
         (133, "7405", 2.20, 1.1, "OF3"), (148, "7410", 2.40, 0.8, "HX"),
         (166, "7400", 1.60, 1.0, "BU"), (169, "7404", 4.60, 1.0, "BU"), (172, "7408", 1.30, 1.0, "BU"), (175, "7401", 1.50, 1.0, "BU"), (178, "7410", 3.40, 1.0, "BU"),
         (181, "7406", 4.20, 1.1, "EF"), (194, "7407", 0.95, 1.0, "OF2"), (203, "7402", 2.00, 1.0, "OF1"),
         (227, "7400", 0.40, 1.0, "BU"), (231, "7410", 1.10, 1.0, "BU"), (234, "7404", 6.00, 1.0, "HX"), (243, "7408", 0.60, 1.0, "BU"),
         (246, "7403", 3.00, 1.1, "EF"), (262, "7402", 4.00, 0.9, "OF2")]
BU_SCALE = [1.0, 1.1, 1.04, 1.13, 1.0, 1.07, 1.12, 1.02, 1.09, 1.0, 1.06, 1.14, 1.03, 1.1]     # irregular punch sizes in bursts
starts = [s[0] for s in SHOTS]
shot_of = np.zeros(N, int)
for i, f0 in enumerate(starts): shot_of[f0:] = i
OUT = f"{E}/out"; os.makedirs(f"{OUT}/frames", exist_ok=True); os.makedirs(f"{E}/build/masks", exist_ok=True)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- source frames
caps = {}; pos = {}; last = {}
def src_frame(clip, t):
    """Frame at time t; reads forward when it can (seeking in long-GOP H.264 decodes from the keyframe every time)."""
    c = caps.get(clip)
    if c is None: c = caps[clip] = cv2.VideoCapture(f"{E}/prox/{clip}.mp4"); pos[clip] = 0
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); i = int(min(n - 1, max(0, round(t * FPS))))
    if clip in last and last[clip][0] == i: return last[clip][1]
    if not (pos[clip] <= i <= pos[clip] + 12):
        c.set(cv2.CAP_PROP_POS_FRAMES, i); pos[clip] = i
    while pos[clip] < i: c.grab(); pos[clip] += 1
    ok, f = c.read(); pos[clip] += 1
    f = f if ok else np.zeros((H, W, 3), np.uint8); last[clip] = (i, f); return f

def shot_time(i, f):
    _, clip, t0, sp, _ = SHOTS[i]; return clip, t0 + (f - starts[i]) / FPS * sp

# ---------------------------------------------------------------- privacy: blur all screen text
def privacy(img, clip):
    if clip in TEXT_CLOSEUP:
        return cv2.GaussianBlur(img, (0, 0), 26)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    scr = ((hsv[..., 2] > 105) & (hsv[..., 1] < 125)).astype(np.uint8) * 255
    scr = cv2.morphologyEx(scr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(scr)
    keep = np.zeros_like(scr)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] > 0.004 * W * H: keep[lab == k] = 255
    if not keep.any(): return img
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    m = cv2.GaussianBlur(keep, (0, 0), 9).astype(np.float32)[..., None] / 255
    b = cv2.GaussianBlur(img, (0, 0), 11)
    return (img * (1 - m) + b * m).astype(np.uint8)

# ---------------------------------------------------------------- smooth, solid subject masks
def clean_mask(m, img, clip):
    b = (m > 0.45).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(b)
    if n < 2:
        b = np.zeros_like(b)
    else:
        big = st[1:, cv2.CC_STAT_AREA].max(); b = np.zeros_like(b)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] >= 0.18 * big: b[lab == k] = 1
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    ff = b.copy(); cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1); b = b | (1 - ff)          # fill holes
    if clip in PERSON:
        # the person, not the screen: take the bright screen out of the mask first, then keep the darkest sizeable part
        # (the silhouette) and grow a body under it, so the white flicker is a person's shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        scr = ((hsv[..., 2] > 95) & (hsv[..., 1] < 140)).astype(np.uint8)
        scr = cv2.dilate(cv2.morphologyEx(scr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))), cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
        b = b & (1 - scr)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n, lab, st, cen = cv2.connectedComponentsWithStats(b)
        best = None
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < 0.01 * W * H: continue
            reg = lab == k; dark = (g[reg] < 60).mean()
            if dark > 0.35 and (best is None or dark * st[k, cv2.CC_STAT_AREA] > best[0]): best = (dark * st[k, cv2.CC_STAT_AREA], k)
        if best is None:                                            # no dark part found: the darkest columns in the lower half
            col = (g[H // 2:] < 45).mean(0); xs = np.where(col > 0.5)[0]
            if len(xs) > W * 0.12:
                x0, x1 = xs.min(), xs.max(); cy = int(H * 0.55); body = np.zeros_like(b)
                cv2.ellipse(body, ((x0 + x1) // 2, cy), (max(40, (x1 - x0) // 5), int(H * 0.06)), 0, 0, 360, 1, -1)
                pts = np.array([[(x0 + x1) // 2 - (x1 - x0) // 4, cy], [(x0 + x1) // 2 + (x1 - x0) // 4, cy], [x1, H], [x0, H]], np.int32)
                cv2.fillPoly(body, [pts], 1); b = b | body
        else:
            k = best[1]; x, y, w, h = st[k, :4]; cx = int(cen[k][0]); body = np.zeros_like(b)
            top = y + int(h * 0.55); hw = max(w // 2, int(W * 0.07))
            pts = np.array([[cx - int(hw * 0.9), top], [cx + int(hw * 0.9), top], [min(W, cx + int(hw * 1.9)), H], [max(0, cx - int(hw * 1.9)), H]], np.int32)
            cv2.fillPoly(body, [pts], 1); b = b | body
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91)))
    else:
        # objects (the fan): the convex hull of the main shape, so its outline has no bites
        n, lab, st, _ = cv2.connectedComponentsWithStats(b)
        if n > 1:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            b = np.zeros_like(b); cv2.fillPoly(b, [cv2.convexHull(max(cs, key=cv2.contourArea))], 1)
    b = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return cv2.GaussianBlur(b.astype(np.float32), (0, 0), 7)

def mask_for(clip, t):
    p = f"{E}/build/masks/{clip}_{t:.3f}.npy"
    if os.path.exists(p): return np.load(p).astype(np.float32)
    img = src_frame(clip, t); m = clean_mask(seg(img), img, clip)
    np.save(p, m.astype(np.float16)); return m

# ---------------------------------------------------------------- camera moves and grade
def transform(img, s, rot, dx, dy, border=cv2.BORDER_REFLECT):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, s); M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border)

def cam(i, f):
    """No shake: a clean punch-in at the cut that settles, a slow push on holds; burst shots alternate two scales."""
    k = f - starts[i]; end = starts[i + 1] if i + 1 < len(SHOTS) else N; L = max(1, end - starts[i]); u = k / L
    typ = SHOTS[i][4]
    if typ == "BU": return BU_SCALE[i % len(BU_SCALE)], 0.0, 0.0, 0.0
    ease = 1 - (1 - min(1, k / 6)) ** 3
    return 1.0 + 0.08 * (1 - ease) + 0.04 * u, 0.0, 0.0, 0.0

def grade(img, f):
    x = img.astype(np.float32) / 255
    x = np.power(x, 0.88)                                              # a little more of what is in the dark
    x = 0.5 + (x - 0.5) * 1.12                                         # contrast
    l = x.mean(2, keepdims=True)
    x = l + (x - l) * 0.90                                             # the red light a touch less saturated
    sh = np.clip(1 - l * 2.2, 0, 1) ** 2
    x += sh * np.array([0.030, 0.012, -0.010], np.float32)             # cool (teal) shadows, BGR
    hi = np.clip((l - 0.62) / 0.38, 0, 1)
    x += hi * np.array([-0.01, 0.0, 0.015], np.float32)                # warm highlights
    br = cv2.GaussianBlur(np.clip(x - 0.6, 0, 1), (0, 0), 28); x += br * 0.45      # bloom
    yy, xx = np.mgrid[0:H, 0:W]; r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    x *= (1 - 0.32 * np.clip(r - 0.45, 0, 1) ** 1.6)[..., None]        # vignette
    x += rng.normal(0, 0.016, (H // 2, W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)     # grain
    return np.clip(x, 0, 1)

VIG = None

# ---------------------------------------------------------------- render
events = []                                                            # (time, kind) for the shutter sounds
demo = "--demo" in sys.argv
OF_cuts = {i for i, s in enumerate(SHOTS) if s[4].startswith("OF") and i > 0}
FLICK = {i: {"OF1": (0,), "OF2": (0, 2), "OF3": (0, 2, 4)}[SHOTS[i][4]] for i in OF_cuts}
for f in range(N):
    i = shot_of[f]; clip, t = shot_time(i, f)
    s, rot, dx, dy = cam(i, f)
    img = privacy(src_frame(clip, t), clip)
    x = grade(transform(img, s, rot, dx, dy), f)
    k_in = f - starts[i]; nxt = i + 1 if i + 1 < len(SHOTS) else None
    k_out = (starts[nxt] - f) if nxt is not None else 99                # frames until the next cut
    # --- the outgoing side of an object flash: subject in colour, background to a hard black/white, then white
    if nxt is not None and nxt in OF_cuts and k_out <= 3:
        m = transform(mask_for(clip, shot_time(i, starts[nxt] - 2)[1]), s, rot, dx, dy, cv2.BORDER_CONSTANT)[..., None]
        l = x.mean(2); q = {3: 0.82, 2: 0.30, 1: 0.0}[k_out]
        thr = np.quantile(l, q) if q > 0 else -1
        bw = cv2.GaussianBlur((l > thr).astype(np.float32), (0, 0), 1.2)[..., None].repeat(3, 2)
        subj = x * 1.15 if k_out > 1 else x * 0.35 + 0.65
        x = np.clip(subj * m + bw * (1 - m), 0, 1)
        if k_out == 3: events.append((f / FPS, "of_out"))
    # --- the incoming side: the subject flickers as a white silhouette (frames 0 and 2)
    if i in OF_cuts and k_in in FLICK[i]:
        m = transform(mask_for(clip, shot_time(i, starts[i] + 1)[1]), s, rot, dx, dy, cv2.BORDER_CONSTANT)[..., None]
        a = 0.92 if k_in == 0 else (0.82 if k_in == 2 else 0.7)
        x = np.clip(x * (1.25 if k_in == 0 else 1.1) * (1 - m * a) + m * a, 0, 1)
        events.append((f / FPS, "of_in" if k_in == 0 else "of_flick"))
    if i in OF_cuts and k_in in tuple(k + 1 for k in FLICK[i]):
        x = np.clip(x * 1.12, 0, 1)
    # --- exposure flash into the shot
    if SHOTS[i][4] == "EF" and k_in < 4:
        a = [1.0, 0.62, 0.32, 0.12][k_in]; x = np.clip(x + a * (1.15 - x) * 0.95, 0, 1)
        if k_in == 0: events.append((f / FPS, "ef"))
    if SHOTS[i][4] == "HC" and k_in == 0:                                 # a one-frame black/white blink on the cut
        l = x.mean(2); x = cv2.GaussianBlur((l > np.quantile(l, 0.6)).astype(np.float32), (0, 0), 1.0)[..., None].repeat(3, 2); events.append((f / FPS, "hc"))
    if SHOTS[i][4] == "BU" and k_in == 0: events.append((f / FPS, "burst"))
    # --- the end: the last frames go to black
    if f >= N - 6: x *= (N - 1 - f) / 6
    cv2.imwrite(f"{OUT}/frames/f{f:04d}.jpg", (x * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if f % 30 == 0: print("frame", f, flush=True)
json.dump({"shots": SHOTS, "starts": starts, "events": events}, open(f"{OUT}/edl.json", "w"), indent=1)

# ---------------------------------------------------------------- sound: the track from the drop, a shutter on every flash
SR = 48000; NS = int(SR * N / FPS)
def load(p, s=0.0, d=None):
    c = ["ffmpeg", "-v", "error", "-ss", str(s)] + (["-t", str(d)] if d else []) + ["-i", p, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(c, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()
bus = np.zeros((NS, 2), np.float32)                                   # no music for now (the client): only the shutters
clicks = {k: load(f"{E}/sfx/mixkit-{k}/{k}.wav") for k in ("1430", "1432", "1434", "1133")}
def pk(x): return int(np.argmax(np.abs(x).max(1)))
def put(x, t, g):
    i0 = int(t * SR) - pk(x); i0 = max(0, i0); n = min(len(x), NS - i0)
    if n > 0: bus[i0:i0 + n] += x[:n] * 10 ** (g / 20)
for t, kind in events:
    if kind == "of_in": put(clicks["1430"][: int(0.4 * SR)], t, -1)
    elif kind == "of_flick": put(clicks["1434"], t, -5)
    elif kind == "of_out": put(clicks["1133"], t, -9)
    elif kind == "ef": put(clicks["1432"][: int(0.45 * SR)], t, -4)
    elif kind == "hc": put(clicks["1430"][: int(0.35 * SR)], t, -3)
    elif kind == "burst":                                                 # a camera in burst mode: alternate two clicks, small level changes
        c = clicks["1434"] if int(t * 1000) % 2 else clicks["1133"]; put(c, t, -3 - 2 * (int(t * 97) % 3))
pre = f"{OUT}/audio_pre.wav"
with wave.open(pre, "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(bus / max(1e-6, np.abs(bus).max()) * 0.95, -1, 1) * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", pre, "-af", "alimiter=limit=0.89:level=false,aresample=48000", "-ar", "48000", f"{OUT}/audio.wav"], check=True)
name = "flash_edit_DEMO3.mp4" if demo else "flash_edit_FINAL.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{OUT}/frames/f%04d.jpg", "-i", f"{OUT}/audio.wav", "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", f"{OUT}/{name}"], check=True)
print("done", name, len(events), "flash sounds")
