"""WKC ad (IMG_6576): Prime captions + view switching + two canvas moments + sound per moment.

Edit decisions (all times in SOURCE seconds; the cuts below remove two pauses):
  keep 0.79-8.29, 8.58-13.21, 13.96-end      (jump cuts hidden by view switches + negative frames)
  canvas 1  "complete motion design ... website voor je maken"   services checklist, title on "overnemen"
  canvas 2  "van nul naar honderd"                                counter 0 -> 100 with a rising line
Usage: python3 build_v13.py OUT_DIR
"""
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
import quick_prime as qp  # noqa: E402

LAB = "/home/user/SocialMediaMarketingSkill/lab"
E = f"{LAB}/edit-03"
VIEWS = {v: f"{E}/views/{v}.mp4" for v in ("wide", "medium", "close")}
MASKS = f"{E}/views/masks"                 # m{source frame + 1}.png for the canvas frames
W, H, FPS = 1080, 1920, 24
KEEP = [(19, 199), (206, 317), (335, 370)]  # source frame ranges [start, end)
OUT_MAP = [f for a, b in KEEP for f in range(a, b)]   # output frame -> source frame
N = len(OUT_MAP)
LEAD = 1                                    # captions appear one frame before the word


def out_frame(src_t):
    """Output frame showing source time src_t (the caption frame), for times inside a kept range."""
    sf = int(np.ceil(src_t * FPS - 1e-6))
    for k, f in enumerate(OUT_MAP):
        if f >= sf:
            return k
    return N - 1


# (text, source start of first word, class, fill)
CAPTIONS = [
    ("bij", 0.881, "sans", "W"), ("WKC", 1.042, "script", "P"), ("kunnen we je", 1.663, "sans", "W"),
    ("complete", 2.143, "sans", "W"), ("motion design", 2.604, "sans", "W"), ("overnemen", 3.686, "sans", "W"),
    ("complete", 4.227, "sans", "W"), ("marketing", 4.667, "sans", "W"), ("complete", 5.188, "sans", "W"),
    ("social media", 5.729, "sans", "W"), ("we kunnen je", 6.330, "sans", "W"), ("complete", 6.851, "sans", "W"),
    ("Branding", 7.392, "script", "W"), ("maken", 7.932, "sans", "W"), ("we kunnen een", 8.600, "sans", "W"),
    ("website", 8.954, "sans", "W"), ("voor je maken", 9.355, "sans", "W"), ("we kunnen een", 10.016, "sans", "W"),
    ("bedrijf", 10.496, "sans", "W"), ("van", 10.917, "sans", "W"), ("0", 11.197, "sans", "W"),
    ("naar", 11.578, "sans", "W"), ("100", 12.239, "sans", "W"), ("Opbouwen", 12.680, "script", "W"),
    ("neem", 14.062, "sans", "W"), ("Contact", 14.282, "script", "W"), ("met ons", 14.743, "sans", "W"),
]
CAP = [dict(text=t, frame=max(0, out_frame(s) - LEAD), cls=c, fill=qp.PINK if f == "P" else qp.WHITE) for t, s, c, f in CAPTIONS]


def cap_frame(text, nth=0):
    return [c["frame"] for c in CAP if c["text"] == text][nth]


CUT_A, CUT_B = OUT_MAP.index(206), OUT_MAP.index(335)
C1 = (cap_frame("complete", 0), cap_frame("we kunnen een", 1))
C2 = (cap_frame("van"), cap_frame("Opbouwen"))
SEGMENTS = [  # (first output frame, view)
    (0, "wide"), (cap_frame("WKC"), "close"), (cap_frame("kunnen we je"), "medium"), (C1[0], "canvas1"),
    (C1[1], "close"), (cap_frame("bedrijf"), "wide"), (C2[0], "canvas2"), (C2[1], "close"),
    (CUT_B, "medium"), (cap_frame("Contact"), "close"), (cap_frame("met ons"), "wide"),
]
NEG = set()
for f, n in [(cap_frame("WKC"), 2), (cap_frame("kunnen we je"), 1), (C1[0], 1), (CUT_A, 1), (C1[1], 1),
             (cap_frame("bedrijf"), 2), (C2[0], 1), (C2[1], 1), (CUT_B, 1), (cap_frame("Contact"), 2), (cap_frame("met ons"), 1)]:
    NEG.update(range(f, f + n))
ROWS = [("Motion design", "motion design"), ("Marketing", "marketing"), ("Social media", "social media"),
        ("Branding", "Branding"), ("Website", "website")]
ROW_POP = [cap_frame(w) + 2 for _, w in ROWS]          # pop 2 frames after the caption: clicks don't collide with shutters
TITLE1_AT = cap_frame("overnemen")
TITLE2_AT = C2[0] + 3
COUNT_FROM, COUNT_TO = cap_frame("naar"), cap_frame("100")
ZERO_AT = cap_frame("0")
SCRIPT_ZOOM = {c["frame"] for c in CAP if c["cls"] == "script"}

BG, DOT, CARD, BORDER, RED = (10, 10, 11), (30, 30, 33), (18, 18, 20), (232, 232, 232), (229, 57, 53)
PANEL = dict(x0=135, x1=945, top=1390, radius=46, scale=0.75, head_top=1310)
CAPTION_Y_CANVAS = 1245


def font(px, wght=800):
    f = ImageFont.truetype(f"{LAB}/caption-style-01/fonts/inter/Inter-Variable.ttf", px, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([32, wght])
    return f


def view_at(i):
    v = SEGMENTS[0][1]
    for f0, name in SEGMENTS:
        if i >= f0:
            v = name
    return v


def read_frames(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def background():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for y in range(18, H, 36):
        for x in range(18, W, 36):
            d.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=DOT)
    return im


def glitch_text(im, lines, i, at, top, px=84):
    """Title that glitches in (3 frames of channel offsets and slice shifts), then sits clean."""
    if i < at:
        return im
    layer = Image.new("L", (W, 260), 0)
    ld = ImageDraw.Draw(layer)
    for li, line in enumerate(lines):
        ld.text((W / 2, 70 + li * 100), line, font=font(px), fill=255, anchor="mm")
    arr = np.asarray(layer).astype(np.float32) / 255
    rgb = np.repeat(arr[..., None], 3, axis=2)
    g = i - at
    if g < 3:
        off = [14, 0, -14] if g != 1 else [-9, 5, 11]
        for c in range(3):
            rgb[..., c] = np.roll(arr, off[c], axis=1)
        for s in range(0, 260, 26):
            if (s // 26 + g) % 3 == 0:
                rgb[s:s + 26] = np.roll(rgb[s:s + 26], 24 * (1 if g % 2 else -1), axis=1)
    base = np.asarray(im).astype(np.float32) / 255
    base[top:top + 260] = np.maximum(base[top:top + 260], rgb)
    return Image.fromarray((base * 255).astype(np.uint8))


def canvas1(i, bg):
    im = glitch_text(bg.copy(), ["Wij nemen het", "over"], i, TITLE1_AT, 190)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 120, 480, 960, 1150
    d.rounded_rectangle((x0, y0, x1, y1), radius=30, fill=CARD, outline=BORDER, width=3)
    newest = max([k for k, f in enumerate(ROW_POP) if i >= f], default=-1)
    for k, (label, _) in enumerate(ROWS):
        if i < ROW_POP[k]:
            continue
        age = i - ROW_POP[k]
        s = 0.7 + 0.3 * min(1, age / 2)
        ry = y0 + 85 + k * 122
        dim = 255 if k == newest else 150                 # focus on the newest item
        r = 26 * s
        d.ellipse((x0 + 70 - r, ry - r, x0 + 70 + r, ry + r), fill=RED if k == newest else (120, 40, 38))
        d.line((x0 + 58, ry, x0 + 68, ry + 10, x0 + 84, ry - 10), fill=(255, 255, 255), width=5)
        d.text((x0 + 120, ry), label, font=font(int(62 * s), 750), fill=(dim, dim, dim), anchor="lm")
    return im


def canvas2(i, bg):
    im = glitch_text(bg.copy(), ["Van 0 naar 100"], i, TITLE2_AT, 230)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 120, 480, 960, 1150
    d.rounded_rectangle((x0, y0, x1, y1), radius=30, fill=CARD, outline=BORDER, width=3)
    if i < COUNT_FROM:
        p = 0.0
    else:
        p = min(1.0, (i - COUNT_FROM) / max(1, COUNT_TO - COUNT_FROM)) ** 2.2   # accelerate into 100
    value = int(round(100 * p))
    if i >= ZERO_AT or value > 0:
        d.text(((x0 + x1) / 2, y0 + 190), str(value), font=font(220, 850), fill=RED if value == 100 else (255, 255, 255), anchor="mm")
    # rising line chart along the bottom of the card
    cx0, cx1, cby, cty = x0 + 60, x1 - 60, y1 - 60, y0 + 360
    d.line((cx0, cby, cx1, cby), fill=(50, 50, 54), width=2)
    pts = []
    for k in range(41):
        q = k / 40
        if q > p + 1e-9:
            break
        wob = 0.05 * np.sin(q * 9)
        pts.append((cx0 + q * (cx1 - cx0), cby - (q ** 1.8 + wob * q) * (cby - cty)))
    if len(pts) > 1:
        d.line(pts, fill=RED, width=7, joint="curve")
        d.ellipse((pts[-1][0] - 13, pts[-1][1] - 13, pts[-1][0] + 13, pts[-1][1] + 13), fill=RED)
    return im


def head_top_px():
    tops = []
    for f in range(46, 312, 10):
        m = np.asarray(Image.open(f"{MASKS}/m{f:05d}.png")) > 128
        rows = np.where(m[:, 200:880].any(axis=1))[0]
        if len(rows):
            tops.append(rows[0])
    return int(np.median(tops))


def panel(canvas_im, frame_rgb, mask_l, head_top):
    s = PANEL["scale"]
    fw, fh = int(W * s), int(H * s)
    small = Image.fromarray(frame_rgb).resize((fw, fh), Image.LANCZOS)
    msmall = Image.fromarray(mask_l).resize((fw, fh), Image.BILINEAR)
    y_off, x_off = int(PANEL["head_top"] - head_top * s), (W - fw) // 2
    layer = Image.new("RGB", (W, H))
    layer.paste(small, (x_off, y_off))
    mlayer = Image.new("L", (W, H), 0)
    mlayer.paste(msmall, (x_off, y_off))
    box = (PANEL["x0"], PANEL["top"], PANEL["x1"], H + PANEL["radius"])
    pmask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(pmask).rounded_rectangle(box, radius=PANEL["radius"], fill=255)
    above = np.zeros((H, W), np.float32)
    above[: PANEL["top"] + 2] = 1
    pop = Image.fromarray((np.asarray(mlayer, np.float32) * above).astype(np.uint8))
    out = canvas_im.copy()
    out.paste(layer, (0, 0), pmask)
    ImageDraw.Draw(out).rounded_rectangle(box, radius=PANEL["radius"], outline=BORDER, width=3)
    out.paste(layer, (0, 0), pop)
    return out


def main(out_dir):
    alphas = [qp.word_alpha(c["text"], c["cls"]) for c in CAP]
    views = {v: read_frames(p) for v, p in VIEWS.items()}
    bg, ht = background(), head_top_px()
    out_path = f"{out_dir}/v13_noaudio.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-vf", "scale=out_color_matrix=bt709", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    for i in range(N):
        sf, v = OUT_MAP[i], view_at(i)
        canvas = v.startswith("canvas")
        if canvas:
            m = np.asarray(Image.open(f"{MASKS}/m{sf + 1:05d}.png"))
            pic = panel((canvas1 if v == "canvas1" else canvas2)(i, bg), views["wide"][sf], m, ht)
            base = np.asarray(pic).astype(np.float32) / 255
        else:
            base = views[v][sf].astype(np.float32) / 255
        zoom = 1.0
        cur = [k for k, c in enumerate(CAP) if c["frame"] <= i]
        if cur:
            c, a = CAP[cur[-1]], alphas[cur[-1]]
            h, w = a.shape
            yc = CAPTION_Y_CANVAS if canvas else H // 2
            x0, y0 = (W - w) // 2, int(yc - h / 2)
            ys, xs = slice(max(0, y0), min(H, y0 + h)), slice(max(0, x0), min(W, x0 + w))
            aa = a[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0][..., None]
            region = base[ys, xs]
            base[ys, xs] = region * (1 - aa) + np.abs(np.array(c["fill"], np.float32) - region) * aa
            if c["cls"] == "script" and not canvas:
                zoom = 1.0 + qp.ZOOM_AMOUNT * (1.0 - np.exp(-(i - c["frame"]) / qp.ZOOM_TAU))
        if i in NEG:
            base = 1.0 - base
        frame = Image.fromarray((np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8))
        if zoom > 1.0005:
            cw, ch = W / zoom, H / zoom
            frame = frame.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()

    t = lambda f: f / FPS  # noqa: E731
    ev = [{"t": t(c["frame"]), "type": "shutter"} for c in CAP]
    ev += [{"t": t(c["frame"]), "type": "hit", "size": 1.2 if c["text"] in ("Opbouwen", "Contact") else 0.9} for c in CAP if c["cls"] == "script"]
    ev += [{"t": t(f), "type": "flash"} for f in (cap_frame("WKC"), cap_frame("kunnen we je"), CUT_A, cap_frame("bedrijf"), CUT_B,
                                                  cap_frame("Contact"), cap_frame("met ons"))]
    ev += [{"t": t(f), "type": "whoosh"} for f in (C1[0], C1[1], C2[0], C2[1])]
    ev += [{"t": t(f), "type": "glitch"} for f in (TITLE1_AT, TITLE2_AT, COUNT_TO)]
    ev += [{"t": t(f), "type": "click"} for f in ROW_POP]
    ev.append({"t": t(COUNT_TO), "type": "riser", "length": t(COUNT_TO - ZERO_AT)})
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{out_dir}/v13_events.json", "w"), indent=1)
    json.dump([{"text": c["text"], "frame": c["frame"], "class": c["cls"]} for c in CAP], open(f"{out_dir}/v13_captions.json", "w"), indent=1)
    print(f"rendered {N} frames ({N/FPS:.2f} s) -> {out_path}; {len(ev)} sound events; head top {ht}px")
    print("canvas1", C1, "canvas2", C2, "cuts", CUT_A, CUT_B)


if __name__ == "__main__":
    main(sys.argv[1])
