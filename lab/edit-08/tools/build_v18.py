"""Savery 1698 -> WKC story edit (clips 7001-7007, filmed at 24 fps: the edit runs at 24), version 1.

Picture per output frame comes from one of: the talking head (wide/medium/close, furnace grade: the client's red
light pushed toward firelight), a real period image (slow camera move), or a canvas moment: a motion-design animation
of how Savery's engine moved water (steam in, cooled to a vacuum, water drawn up from the mine, steam pushes it out),
and a tangle that straightens into one clear line ("complexity easier to understand"), with the speaker in the
bottom panel and the head popping out over its edge. Prime captions on top. Pacing from out/edl.json, decisions from
out/edit_v18.json.
Usage: python3 build_v18.py [OUT.mp4]
"""
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/edit-05/tools")
sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/edit-08/tools")
import quick_prime as qp  # noqa: E402
from grade import grade_print  # noqa: E402
from grade_furnace import grade_furnace as grade_talking  # noqa: E402  (red light read as furnace glow)

Image.MAX_IMAGE_PIXELS = None
LAB = "/home/user/SocialMediaMarketingSkill/lab"
E = f"{LAB}/edit-08"
W, H, FPS = 1080, 1920, 24
EDL = json.load(open(f"{E}/out/edl.json"))
CFG = json.load(open(f"{E}/out/edit_v18.json"))
SPEECH_END = EDL["duration"]
N = int(round((SPEECH_END + CFG["end_card_s"]) * FPS))
rng = np.random.default_rng(7)
PUNCH = 0.04                      # client: don't push far into the face
NOW_CLIPS = ("7006", "7007")      # the closing part gets the 'now' grade
CLIPS = ("7001", "7003", "7004", "7005", "7006", "7007")

NAVY, DOT, CARD, BORDER = (9, 13, 26), (24, 32, 54), (16, 22, 40), (226, 228, 235)
BLUE, INK, MUTED = (63, 140, 232), (240, 238, 232), (140, 146, 165)   # BLUE: the WKC site's button colour
PANEL = dict(x0=135, x1=945, top=1390, radius=46, scale=0.75, head_top=1310)
CAPTION_Y_CANVAS = 1245


def font(px, wght=800):
    f = ImageFont.truetype(f"{LAB}/caption-style-01/fonts/inter/Inter-Variable.ttf", px, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([32, wght])
    return f


def seg_at(t):
    for s in EDL["segments"]:
        if s["out_start"] - 1e-6 <= t < s["out_end"] - 1e-6:
            return s
    return EDL["segments"][-1]


def window(lst, t):
    for it in lst:
        if it["start"] - 1e-6 <= t < it["end"] - 1e-6:
            return it
    return None


class Reader:
    """Streams frames forward from a video (decoding on demand keeps memory flat); restarts on a backward seek."""

    def __init__(self, path):
        self.path, self.proc, self.idx, self.last = path, None, -1, None

    def _open(self):
        if self.proc:
            self.proc.kill()
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", self.path, "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709",
                                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.idx, self.last = -1, None

    def get(self, k):
        if self.proc is None or k < self.idx:
            self._open()
        while self.idx < k:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                return self.last                     # past the end: hold the last frame
            self.idx += 1
            self.last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        return self.last


# ---------------------------------------------------------------- prints
class PrintShot:
    """A still print on screen with a slow camera move.
    fill: a 9:16 window around the subject (move 'push' = 6% push-in, 'rise' = the image drifts up so the balloon
          climbs, 'down' = drifts down). Used when the subject fits a 9:16 window.
    fit:  the whole print across the width over a dark, blurred copy of itself, slow 5% push-in."""

    def __init__(self, ins):
        im = Image.open(ins["file"]).convert("RGB")
        self.colour = bool(ins.get("colour"))
        self.dur = ins["end"] - ins["start"]
        self.move = ins.get("move", "push")
        iw, ih = im.size
        x0, y0, x1, y1 = ins.get("subject_box") or [0, 0, 1, 1]
        if ins.get("center"):                                            # explicit framing centre (normalised)
            x0 = x1 = ins["center"][0]
            y0 = y1 = ins["center"][1]
        win_h = min(ih, iw * 16 / 9)
        win_w = win_h * 9 / 16
        self.mode = ins.get("mode") or ("fill" if (x1 - x0) * iw <= win_w * 1.02 else "fit")
        if self.mode == "fill":
            win_h *= ins.get("zoom", 1.0)                                # tighter window on the subject
            win_w = win_h * 9 / 16
            # downscale so the window is ~1.3x the output height (sharp, but cheap to resample every frame)
            k = min(1.0, 1.3 * H / win_h)
            self.im = im.resize((max(1, int(iw * k)), max(1, int(ih * k))), Image.LANCZOS)
            self.win = (win_w * k, win_h * k)
            self.cx = float(np.clip((x0 + x1) / 2 * iw * k, win_w * k / 2, iw * k - win_w * k / 2))
            self.cy = float(np.clip((y0 + y1) / 2 * ih * k, win_h * k / 2, ih * k - win_h * k / 2))
        else:
            fit = im.resize((W, int(ih * W / iw)), Image.LANCZOS) if ih * W / iw <= H * 0.86 else \
                im.resize((int(iw * H * 0.86 / ih), int(H * 0.86)), Image.LANCZOS)
            bgk = max(W / iw, H / ih)
            bg = im.resize((int(iw * bgk) + 1, int(ih * bgk) + 1), Image.BILINEAR).crop((0, 0, W, H))
            bg = bg.filter(ImageFilter.GaussianBlur(28))
            bg = Image.fromarray((np.asarray(bg).astype(np.float32) * 0.32).astype(np.uint8))
            comp = bg.copy()
            comp.paste(fit, ((W - fit.width) // 2, (H - fit.height) // 2))
            self.im = comp.resize((int(W * 1.3), int(H * 1.3)), Image.LANCZOS)
            self.win = (W * 1.3, H * 1.3)
            self.cx, self.cy = W * 1.3 / 2, H * 1.3 / 2

    def frame(self, t):
        p = np.clip(t / max(0.1, self.dur), 0, 1)
        p = p * p * (3 - 2 * p) * 0.35 + p * 0.65                       # mostly linear, soft ends
        ww, wh = self.win
        if self.mode == "fit":
            z = 1 / (1 + 0.05 * p)
            cx, cy = self.cx, self.cy
        elif self.move == "rise":
            z = 0.90
            travel = 0.10 * wh
            cx, cy = self.cx, self.cy - travel / 2 + travel * p     # window moves down -> the balloon climbs
        elif self.move == "spin":                                   # a round chart turning slowly
            z = 0.78
            cx, cy = self.cx, self.cy
        elif self.move == "down":
            z = 0.90
            travel = 0.10 * wh
            cx, cy = self.cx, self.cy + travel / 2 - travel * p
        else:
            z = 1 / (1 + 0.06 * p)
            cx, cy = self.cx, self.cy
        cw, ch = ww * z, wh * z
        cx = float(np.clip(cx, cw / 2, self.im.width - cw / 2))
        cy = float(np.clip(cy, ch / 2, self.im.height - ch / 2))
        if self.move == "spin":
            # crop 20% wider, rotate, crop back: no black corners (6 degrees on 9:16 needs ~16% margin)
            m = 1.2
            big = self.im.resize((int(W * m), int(H * m)), Image.BICUBIC,
                                 box=(cx - cw * m / 2, cy - ch * m / 2, cx + cw * m / 2, cy + ch * m / 2))
            big = big.rotate(-6 * p, resample=Image.BICUBIC)
            out = big.crop(((big.width - W) // 2, (big.height - H) // 2, (big.width - W) // 2 + W, (big.height - H) // 2 + H))
            return np.asarray(out).astype(np.float32) / 255
        out = self.im.resize((W, H), Image.BICUBIC, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))
        return np.asarray(out).astype(np.float32) / 255


class VideoShot:
    """Real footage (e.g. NASA) filling the 9:16 frame (scaled to cover, centre crop), from src seconds in."""

    def __init__(self, ins):
        self.colour = True
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(ins.get("src", 0.0)), "-i", ins["file"], "-vf",
                                      f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                                      "scale=in_color_matrix=bt709:out_color_matrix=bt709", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                     stdout=subprocess.PIPE)
        self.last, self.n = None, 0

    def frame(self, t):
        while self.n <= int(round(t * FPS)):
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            self.last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.n += 1
        return self.last.astype(np.float32) / 255


# ---------------------------------------------------------------- canvases
def background():
    im = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(im)
    for y in range(18, H, 36):
        for x in range(18, W, 36):
            d.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=DOT)
    return im


def glitch_text(im, lines, i, at, top, px=84, colour=(255, 255, 255)):
    """Title that glitches in (3 frames of channel offsets and slice shifts), then sits clean."""
    if i < at:
        return im
    layer = Image.new("L", (W, 120 + 100 * len(lines)), 0)
    ld = ImageDraw.Draw(layer)
    for li, line in enumerate(lines):
        ld.text((W / 2, 70 + li * 100), line, font=font(px), fill=255, anchor="mm")
    arr = np.asarray(layer).astype(np.float32) / 255
    lh = arr.shape[0]
    rgb = np.repeat(arr[..., None], 3, axis=2)
    g = i - at
    if g < 3:
        off = [14, 0, -14] if g != 1 else [-9, 5, 11]
        for c in range(3):
            rgb[..., c] = np.roll(arr, off[c], axis=1)
        for s in range(0, lh, 26):
            if (s // 26 + g) % 3 == 0:
                rgb[s:s + 26] = np.roll(rgb[s:s + 26], 24 * (1 if g % 2 else -1), axis=1)
    rgb = rgb * (np.array(colour, np.float32) / 255)
    base = np.asarray(im).astype(np.float32) / 255
    base[top:top + lh] = np.maximum(base[top:top + lh], rgb)
    return Image.fromarray((base * 255).astype(np.uint8))


def pop(age, frames=3):
    """Scale for an element popping in: 0.7 -> 1.06 -> 1.0."""
    if age < 0:
        return 0
    a = min(1.0, age / frames)
    return 0.7 + 0.36 * a - 0.06 * max(0.0, (age - frames) / 3) if age < frames + 3 else 1.0


PIPE, STEAM, WATER, FIRE, MUTE = (150, 158, 178), (238, 240, 245), (63, 140, 232), (255, 140, 40), (118, 124, 145)


def _ease(g):
    g = min(1.0, max(0.0, g))
    return g * g * (3 - 2 * g)


def _path_len(pts):
    return sum(np.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1))


def _partial(pts, g):
    """The first fraction g of a polyline (for pipes drawing in and water running along them)."""
    if g >= 1:
        return pts
    total, out, run = _path_len(pts) * g, [pts[0]], 0.0
    for k in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        seg = np.hypot(x1 - x0, y1 - y0)
        if run + seg >= total:
            f = (total - run) / max(seg, 1e-6)
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
            return out
        out.append((x1, y1))
        run += seg
    return out


# Savery's engine, simplified to its working parts (boiler, steam pipe + valve, receiver vessel, suction pipe from the
# mine water with a non-return valve, delivery pipe up with a non-return valve, outlet) in screen coordinates
STEAM_PIPE = [(220, 690), (220, 520), (560, 520), (560, 572)]
SUCTION = [(560, 812), (560, 1080)]
DELIVERY = [(612, 790), (850, 790), (850, 400), (930, 400)]
VESSEL = (470, 572, 650, 812)


def engine(i, bg):
    """How Savery's engine moved water, step by step on the speaker's words."""
    b = CFG["beats"]
    im = glitch_text(bg.copy(), ["Savery's engine, 1698"], i, b["engine_title"], 150, px=72)
    if i < b["draw"]:
        return im
    d = ImageDraw.Draw(im)
    g = _ease((i - b["draw"]) / 10)
    f = font(30, 650)
    # mine water and the boiler with its fire
    d.rounded_rectangle((420, 1070, 700, 1130), radius=14, fill=(30, 70, 120))
    d.text((560, 1100), "mine water", font=f, fill=INK, anchor="mm")
    flick = 0.8 + 0.2 * np.sin(i * 1.7) * np.sin(i * 0.9)
    for k, dx in enumerate((-60, -20, 20, 60)):
        h = (48 + 14 * np.sin(i * 1.3 + k)) * flick
        d.polygon([(220 + dx - 22, 990), (220 + dx + 22, 990), (220 + dx, 990 - h)], fill=FIRE)
    d.rounded_rectangle((110, 690, 330, 935), radius=40, outline=PIPE, width=6, fill=(26, 30, 44))
    d.text((220, 812), "boiler", font=f, fill=INK, anchor="mm")
    # pipes draw in
    for pts in (STEAM_PIPE, SUCTION, DELIVERY):
        p = _partial(pts, g)
        if len(p) > 1:
            d.line(p, fill=PIPE, width=14, joint="curve")
    d.rounded_rectangle(VESSEL, radius=60, outline=PIPE, width=6, fill=(22, 26, 40))
    if g >= 1:
        d.ellipse((370, 505, 400, 535), outline=INK, width=4)                              # steam valve
        for vx, vy in ((560, 960), (850, 600)):                                            # non-return valves
            d.polygon([(vx - 16, vy + 10), (vx + 16, vy + 10), (vx, vy - 12)], fill=INK)
    x0, y0, x1, y1 = VESSEL
    label, sub = None, None
    # 1. steam in
    if i >= b["steam"]:
        s = _ease((i - b["steam"]) / 12)
        if i >= b["condense"]:
            s *= 1 - _ease((i - b["condense"]) / 5)                                        # 2. cooled: steam turns back to water, a vacuum
        if s > 0.02:
            col = tuple(int(22 + (c - 22) * s * 0.9) for c in STEAM)
            d.rounded_rectangle((x0 + 8, y0 + 8, x1 - 8, y1 - 8), radius=52, fill=col)
            d.line(_partial(STEAM_PIPE, min(1, (i - b["steam"]) / 6)), fill=STEAM, width=8)
        label = "1. Steam fills the vessel"
    if b["condense"] <= i < b["rise"] + 3:
        for k in range(7):                                                               # cold water poured over it
            yy = y0 - 40 + ((i * 9 + k * 23) % 60)
            d.ellipse((x0 + 20 + k * 22, yy, x0 + 28 + k * 22, yy + 12), fill=(150, 200, 255))
        label = "2. Cooled: a vacuum"
    # 3. the vacuum draws water up from the mine
    level = 0.0
    if i >= b["rise"]:
        level = 0.72 * _ease((i - b["rise"]) / 7)
        d.line(SUCTION, fill=WATER, width=8)
        label = "3. Water drawn up from the mine"
    # 4. steam pushes it up and out
    if i >= b["push"]:
        p = _ease((i - b["push"]) / 14)
        level *= 1 - 0.75 * p
        d.rounded_rectangle((x0 + 8, y0 + 8, x1 - 8, y1 - 8 - (y1 - y0 - 16) * level), radius=52, fill=tuple(int(c * 0.9) for c in STEAM))
        d.line(STEAM_PIPE, fill=STEAM, width=8)
        dl = _partial(DELIVERY, p)
        if len(dl) > 1:
            d.line(dl, fill=WATER, width=8)
        if p >= 1:
            for k in range(6):                                                           # pouring out
                yy = 410 + ((i * 11 + k * 31) % 150)
                d.ellipse((935 + (k % 2) * 8, yy, 949 + (k % 2) * 8, yy + 16), fill=WATER)
        label = "4. Steam pushes it up and out"
    if level > 0.01:
        top = y1 - 8 - (y1 - y0 - 16) * level
        d.rounded_rectangle((x0 + 8, top, x1 - 8, y1 - 8), radius=40, fill=WATER)
    if i >= b["purpose"]:                                                            # the purpose replaces the step label
        s = pop(i - b["purpose"])
        fp = font(int(50 * s), 780)
        tx = "Pumping water out of mines"
        bx = d.textbbox((W / 2, 318), tx, font=fp, anchor="mm")
        d.rounded_rectangle((bx[0] - 26, bx[1] - 16, bx[2] + 26, bx[3] + 16), radius=24, fill=BLUE)
        d.text((W / 2, 318), tx, font=fp, fill=(255, 255, 255), anchor="mm")
    elif label:
        d.text((W / 2, 318), label, font=font(52, 740), fill=INK, anchor="mm")
    return im


_trng = np.random.default_rng(5)
_TANGLE = [(540 + 300 * np.sin(k * 0.61 + 0.4 * np.sin(k * 0.23)) * np.cos(k * 0.17),
            660 + 230 * np.sin(k * 0.37 + 1.1) * np.cos(k * 0.29) + _trng.normal(0, 12)) for k in range(70)]
_STRAIGHT = [(170 + 740 * k / 69, 660) for k in range(70)]


def untangle(i, bg):
    """"It makes complexity easier to understand": a tangle that straightens into one clear arrow."""
    b = CFG["beats"]
    im = glitch_text(bg.copy(), ["Motion design"], i, b["untangle_title"], 150, px=86)
    if i < b["tangle"]:
        return im
    d = ImageDraw.Draw(im)
    g = _ease((i - b["straight"]) / 10) if i >= b["straight"] else 0.0
    grow = min(1.0, (i - b["tangle"]) / 8)
    pts = [(tx + (sx - tx) * g, ty + (sy - ty) * g) for (tx, ty), (sx, sy) in zip(_TANGLE, _STRAIGHT)]
    pts = pts[: max(2, int(len(pts) * grow))]
    col = tuple(int(c1 + (c2 - c1) * g) for c1, c2 in zip(INK, BLUE))
    d.line(pts, fill=col, width=9, joint="curve")
    if g > 0.85:
        ex, ey = pts[-1]
        d.polygon([(ex + 6, ey), (ex - 30, ey - 22), (ex - 30, ey + 22)], fill=BLUE)
    word, wy = ("Clear", 980) if g > 0.5 else ("Complex", 980)
    d.text((W / 2, wy), word, font=font(int(96 * (pop(i - b["straight"]) if g > 0.5 else pop(i - b["tangle"]))), 820),
           fill=BLUE if g > 0.5 else INK, anchor="mm")
    return im


def head_top_px(masks):
    tops = []
    for m in masks.values():
        a = np.asarray(m) > 128
        rows = np.where(a[:, 200:880].any(axis=1))[0]
        if len(rows):
            tops.append(rows[0])
    return int(np.median(tops)) if tops else 700


def panel(canvas_im, frame_rgb, mask_l, head_top):
    """Speaker in the bottom panel; the head (subject mask) pops out over the panel's top edge."""
    s = PANEL["scale"]
    fw, fh = int(W * s), int(H * s)
    small = Image.fromarray(frame_rgb).resize((fw, fh), Image.LANCZOS)
    y_off, x_off = int(PANEL["head_top"] - head_top * s), (W - fw) // 2
    layer = Image.new("RGB", (W, H))
    layer.paste(small, (x_off, y_off))
    box = (PANEL["x0"], PANEL["top"], PANEL["x1"], H + PANEL["radius"])
    pmask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(pmask).rounded_rectangle(box, radius=PANEL["radius"], fill=255)
    out = canvas_im.copy()
    out.paste(layer, (0, 0), pmask)
    ImageDraw.Draw(out).rounded_rectangle(box, radius=PANEL["radius"], outline=BORDER, width=3)
    if mask_l is not None:
        msmall = Image.fromarray(mask_l).resize((fw, fh), Image.BILINEAR)
        mlayer = Image.new("L", (W, H), 0)
        mlayer.paste(msmall, (x_off, y_off))
        above = np.zeros((H, W), np.float32)
        above[: PANEL["top"] + 2] = 1
        out.paste(layer, (0, 0), Image.fromarray((np.asarray(mlayer, np.float32) * above).astype(np.uint8)))
    return out


def end_card(i, f0, bg, logo):
    im = bg.copy()
    if logo is not None:
        lg = logo.copy()
        lg.thumbnail((620, 300))
        im.paste(lg, ((W - lg.width) // 2, 760 - lg.height // 2), lg)
        ty = 900
    else:
        ty = 820
    im = glitch_text(im, ["wkconversions.com"], i, f0, ty, px=78)
    d = ImageDraw.Draw(im)
    g = min(1.0, max(0, i - f0 - 3) / 8)
    half = int(260 * (g * g * (3 - 2 * g)))
    if half:
        d.rounded_rectangle((W // 2 - half, ty + 150, W // 2 + half, ty + 156), radius=3, fill=BLUE)
    return im


# ---------------------------------------------------------------- main
def main(out_path):
    caps = CFG["captions"]
    frames = CFG["cap_frames"]
    alphas = [qp.word_alpha(c["text"], c["cls"]) for c in caps]
    views = {(clip, v): Reader(f"{E}/views/{clip}_{v}.mp4") for clip in CLIPS for v in ("wide", "medium", "close")}
    shots = {k: (VideoShot(ins) if ins.get("kind") == "video" else PrintShot(ins)) for k, ins in enumerate(CFG["inserts"])}
    logo = Image.open(CFG["logo"]).convert("RGBA") if CFG.get("logo") else None
    masks = {}
    import glob
    import os
    for p in glob.glob(f"{E}/masks/*/m*.png"):
        masks[(os.path.basename(os.path.dirname(p)), int(os.path.basename(p)[1:6]))] = Image.open(p).convert("L")
    head_top = head_top_px(masks)
    bg = background()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-vf", "scale=out_color_matrix=bt709", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    neg = set(CFG["negative_frames"])
    for i in range(N):
        t = i / FPS
        canvas_on = False
        if t >= SPEECH_END:                                               # end card
            pic = np.asarray(end_card(i, int(round(SPEECH_END * FPS)), bg, logo)).astype(np.float32) / 255
            canvas_on = True
        else:
            seg = seg_at(t)
            sf = int(round((seg["src_start"] + (t - seg["out_start"])) * FPS))
            era = "now" if seg["clip"] in NOW_CLIPS else "history"
            ins = window(CFG["inserts"], t)
            cv = window(CFG["canvases"], t)
            if ins is not None:
                k = CFG["inserts"].index(ins)
                fr = shots[k].frame(t - ins["start"])
                pic = fr if ins.get("kind") == "video" or ins.get("raw") else grade_print(fr, rng, shots[k].colour)
            elif cv is not None:
                frame = views[(seg["clip"], "wide")].get(sf)
                head = grade_talking(frame.astype(np.float32) / 255, era, rng)
                c_im = engine(i, bg) if cv["kind"] == "engine" else untangle(i, bg)
                m = masks.get((seg["clip"], sf))
                pic = np.asarray(panel(c_im, (head * 255).astype(np.uint8), None if m is None else np.asarray(m), head_top)).astype(np.float32) / 255
                canvas_on = True
            else:
                v = window(CFG["views"], t)["view"]
                frame = views[(seg["clip"], v)].get(sf)
                pic = grade_talking(frame.astype(np.float32) / 255, era, rng)
        base = pic
        zoom = 1.0
        cur = [k for k, f in enumerate(frames) if f <= i]
        if cur and t < SPEECH_END:
            k = cur[-1]
            c, a = caps[k], alphas[k]
            h, w = a.shape
            yc = CAPTION_Y_CANVAS if canvas_on else H // 2
            x0, y0 = (W - w) // 2, int(yc - h / 2)
            ys, xs = slice(max(0, y0), min(H, y0 + h)), slice(max(0, x0), min(W, x0 + w))
            aa = a[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0][..., None]
            fill = np.array(qp.PINK if c.get("fill") == "PINK" else qp.WHITE, np.float32)
            region = base[ys, xs]
            base[ys, xs] = region * (1 - aa) + np.abs(fill - region) * aa
            if c["cls"] == "script" and not canvas_on:
                zoom = 1.0 + PUNCH * (1.0 - np.exp(-(i - frames[k]) / qp.ZOOM_TAU))
        if i in neg:
            base = 1.0 - base
        out = Image.fromarray((np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8))
        if zoom > 1.0005:
            cw, ch = W / zoom, H / zoom
            out = out.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
        enc.stdin.write(out.tobytes())
        if i % 90 == 0:
            print(f"  frame {i}/{N}", flush=True)
    enc.stdin.close()
    enc.wait()
    json.dump([{"text": c["text"], "frame": f, "class": c["cls"]} for c, f in zip(caps, frames)], open(f"{E}/out/v18_captions.json", "w"), indent=1)
    print(f"rendered {N} frames ({N / FPS:.2f} s) -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{E}/out/v18_noaudio.mp4")
