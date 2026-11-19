# Is the picture explaining the voice-over, or is the voice talking over a still page? Splits the force-aligned
# voice-over into phrases and measures, on the render, whether something new appears on screen during each one.
#   python3 phrase_check.py plan words.json            -> the phrases with their times: one shot each (the shot list)
#   python3 phrase_check.py check film.mp4 words.json [--sheet phrases.png]
#       -> per phrase: how much of the frame became new while it was spoken; the phrases that got nothing new
#          ("STATIC"), the longest stretch of voice without a new visual, and a sheet of each phrase's frames
# words.json: {"w:word": start_seconds, ...} (vo_align.py output, or a Remotion project's src/words.json).
# Karl's rule (Oct 2026): the visuals show what is being said, phrase by phrase; never a page that holds while the
# voice explains it. Targets: every phrase gets a new visual (6% of the frame changed, or 0.6% of it new structure, while it plays),
# and no more than 2.5 s of voice passes without one. Camera drift and a slow push are not new visuals: the frames
# are aligned for small zooms and shifts before they are compared. Needs numpy, opencv-python-headless, Pillow.
import json, sys
import numpy as np

GAP = 0.22           # a pause this long ends a phrase
MAX_WORDS, MAX_S = 9, 3.2
NEW = 0.06           # share of the frame whose tone must change during a phrase, or
STRUCT = 0.006       # share of the frame that must gain new structure (edges not there before: a card, a chart, a word)
ELEM = 0.0015       # or one new element: a connected shape of new edges whose box covers this share of the frame
                     # (an icon added to a line diagram; measured 2026-10: DemoDuck's diagram builds add one icon per
                     # phrase, too small for NEW and STRUCT, and were wrongly called static)
LONGEST = 2.5        # seconds of voice allowed without a new visual

def phrases(words):
    ws = sorted(((t, k.split(":", 1)[1].rstrip("0123456789") or k) for k, t in words.items() if k.startswith("w:")))
    out, cur = [], []
    for i, (t, w) in enumerate(ws):
        cur.append((t, w))
        nxt = ws[i + 1][0] if i + 1 < len(ws) else None
        est_end = t + 0.32
        if nxt is None or nxt - est_end > GAP or len(cur) >= MAX_WORDS or (nxt - cur[0][0]) > MAX_S:
            end = min(nxt if nxt is not None else t + 0.6, t + 0.9)
            out.append((cur[0][0], end, " ".join(x[1] for x in cur)))
            cur = []
    return out

def frames(path, fps_out=10, w=640):
    import cv2
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / fps_out)); out = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % step == 0:
            g = cv2.cvtColor(cv2.resize(f, (w, int(w * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            out.append(g.astype(np.float32))
        i += 1
    return out, fps / step

def _edges(g):
    import cv2
    return cv2.Canny(g.astype(np.uint8), 40, 110) > 0

def newness(a, b):
    """What is new in b relative to a, after aligning small zooms and shifts (a camera move is not new content):
    (share of the frame whose tone changed, share of the frame that gained new edges). A white card landing on
    white changes little tone but adds a lot of structure; a bullet appearing on a held page adds almost none."""
    import cv2
    tone = struct = elem = None
    for s in (1.0, 1.02, 0.98, 1.04, 0.96, 1.08, 0.93):
        M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), 0, s)
        aa = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            _, warp = cv2.findTransformECC(aa, b, warp, cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4), None, 3)
            aa = cv2.warpAffine(aa, warp, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            pass
        d = np.abs(cv2.GaussianBlur(aa, (5, 5), 0) - cv2.GaussianBlur(b, (5, 5), 0))
        t = float((d > 18).mean())
        ea = cv2.dilate(_edges(aa).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        new = _edges(b) & ~ea; e = float(new.mean())
        # the biggest single new shape: new edges grouped into blobs, the largest blob's box as a share of the frame
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(new.astype(np.uint8), np.ones((9, 9), np.uint8)), 8)
        el = 0.0
        for k in range(1, n):
            if new[lab == k].sum() >= 25: el = max(el, st[k, 2] * st[k, 3] / new.size)
        tone = t if tone is None else min(tone, t); struct = e if struct is None else min(struct, e)
        elem = el if elem is None else min(elem, el)
    return tone, struct, elem

def check(video, words_path, sheet=None):
    ph = phrases(json.load(open(words_path)))
    fr, fps = frames(video)
    at = lambda t: fr[max(0, min(len(fr) - 1, int(round(t * fps))))]
    rows, static = [], []
    for a, b, text in ph:
        # the most that became new between the phrase's start and any moment in it
        f0 = at(a - 0.1)
        vals = [newness(f0, at(t)) for t in np.linspace(a + 0.15, b, 5)]
        tone, struct, elem = max(v[0] for v in vals), max(v[1] for v in vals), max(v[2] for v in vals)
        ok = tone >= NEW or struct >= STRUCT or elem >= ELEM
        rows.append((a, b, ok, tone, struct, text, elem))
        if not ok: static.append(text)
    # the longest stretch of voice without a new visual: walk the phrases, resetting on each that brings one
    longest, run_start, last = 0.0, None, None
    for a, b, ok, *_ in rows:
        if ok:
            if run_start is not None: longest = max(longest, last - run_start)
            run_start = None
        else:
            run_start = a if run_start is None else run_start; last = b
    if run_start is not None: longest = max(longest, last - run_start)
    for a, b, ok, tone, struct, text, elem in rows:
        print(f"{'      ' if ok else 'STATIC'} {a:6.2f}–{b:5.2f}s  tone {tone * 100:5.1f}%  structure {struct * 100:4.1f}%  element {elem * 100:4.2f}%  {text}")
    print(f"{len(rows) - len(static)} of {len(rows)} phrases bring a new visual (target: all; {NEW * 100:.0f}% of the frame changed, {STRUCT * 100:.1f}% new structure, or one new element of {ELEM * 100:.2f}%)")
    print(f"longest voice without a new visual: {longest:.1f} s (target {LONGEST} s or less)")
    if sheet:
        from PIL import Image, ImageDraw
        W, H = 240, 135; cols = 8; rws = (len(rows) + cols - 1) // cols
        im = Image.new("RGB", (cols * W, rws * (H + 30)), "white"); d = ImageDraw.Draw(im)
        for i, (a, b, ok, tone, struct, text, elem) in enumerate(rows):
            f = Image.fromarray(at((a + b) / 2).astype(np.uint8)).resize((W, H)).convert("RGB")
            x, y = (i % cols) * W, (i // cols) * (H + 30); im.paste(f, (x, y + 30))
            d.text((x + 4, y + 2), f"{a:.1f}s {'' if ok else 'STATIC '}{tone * 100:.0f}% / {struct * 100:.1f}%", fill="black" if ok else "red")
            d.text((x + 4, y + 15), text[:38], fill="black")
        im.save(sheet)
    return 1 if static or longest > LONGEST else 0

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "plan":
        for i, (a, b, text) in enumerate(phrases(json.load(open(sys.argv[2])))):
            print(f"{i + 1:02d}  {a:6.2f}–{b:5.2f}s  {text}")
    elif len(sys.argv) >= 4 and sys.argv[1] == "check":
        sh = sys.argv[sys.argv.index("--sheet") + 1] if "--sheet" in sys.argv else None
        sys.exit(check(sys.argv[2], sys.argv[3], sh))
    else:
        print(__doc__ if __doc__ else open(__file__).read().split("\nimport")[0]); sys.exit(2)
