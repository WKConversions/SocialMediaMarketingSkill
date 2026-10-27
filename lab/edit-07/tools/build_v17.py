"""Röntgen 1895 -> WKC story edit (clips 6952-6958, filmed at 24 fps: the edit runs at 24), version 1.

Picture per output frame comes from one of: the talking head (wide/medium/close, graded 'history' for the 1895 part
and 'now' for the closing part), an X-ray scan sweeping over the speaker ("he could see through human skin": a
visual effect, not real bones), a real archival image (slow camera move), or a canvas moment (the 1895 hand X-ray
hidden, then revealed on "visible"; "explain it -> show it") with the speaker in the bottom panel and the head popping
out over its edge. Prime captions on top. Pacing from out/edl.json, decisions from out/edit_v17.json.
Usage: python3 build_v17.py [OUT.mp4]
"""
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/edit-05/tools")
import quick_prime as qp  # noqa: E402
from grade import grade_print  # noqa: E402
from grade_darkroom import grade_darkroom as grade_talking  # noqa: E402  (filmed under red darkroom light)

Image.MAX_IMAGE_PIXELS = None
LAB = "/home/user/SocialMediaMarketingSkill/lab"
E = f"{LAB}/edit-07"
W, H, FPS = 1080, 1920, 24
EDL = json.load(open(f"{E}/out/edl.json"))
CFG = json.load(open(f"{E}/out/edit_v17.json"))
SPEECH_END = EDL["duration"]
N = int(round((SPEECH_END + CFG["end_card_s"]) * FPS))
rng = np.random.default_rng(7)
PUNCH = 0.04                      # client: don't push far into the face
NOW_CLIPS = ("6957", "6958")      # the closing part gets the 'now' grade
CLIPS = ("6952", "6953", "6954", "6955", "6956", "6957", "6958")

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


def xray_look(x):
    """What the scan shows behind its line: the speaker as glowing cyan edges on black, like a radiograph on a lightbox
    (a visual effect, not real bones). The dark footage is smoothed first so its noise doesn't sparkle."""
    from PIL import Image as _I, ImageFilter as _F
    lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    im = _I.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8)).filter(_F.GaussianBlur(2.2))
    sm = np.asarray(im).astype(np.float32) / 255
    blur = np.asarray(im.resize((W // 4, H // 4), _I.BILINEAR).filter(_F.GaussianBlur(6)).resize((W, H), _I.BILINEAR)).astype(np.float32) / 255
    edge = np.clip(sm + 2.0 * (sm - blur), 0, 1)
    v = np.clip((edge - 0.12) * 2.3, 0, 1) ** 1.1
    return np.clip(v[..., None] * np.array([0.42, 0.86, 1.0], np.float32) + v[..., None] ** 3 * 0.25, 0, 1)


def scan_fx(x, p):
    """An X-ray scan line sweeping top to bottom over the speaker; above it the blue negative."""
    yl = (-0.06 + 1.12 * p) * H
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    above = np.clip((yl - yy) / 6.0, 0, 1)
    out = x * (1 - above) + xray_look(x) * above
    band = np.exp(-((yy - yl) / 14.0) ** 2)
    glow = np.exp(-np.abs(yy - yl) / 70.0) * 0.25
    return np.clip(out + (band * 0.85 + glow) * np.array([0.55, 0.85, 1.0], np.float32), 0, 1)


def _card(im, img, box, sharp, dim):
    """The radiograph in a rounded card: sharp in 0..1 (0 = lost in fog), dim in 0..1."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    k = max(w / img.width, h / img.height)
    fit = img.resize((int(img.width * k) + 1, int(img.height * k) + 1), Image.LANCZOS)
    fit = fit.crop(((fit.width - w) // 2, (fit.height - h) // 2, (fit.width - w) // 2 + w, (fit.height - h) // 2 + h))
    if sharp < 1:
        fit = fit.filter(ImageFilter.GaussianBlur(26 * (1 - sharp)))
    a = np.asarray(fit).astype(np.float32) / 255 * (0.18 + 0.82 * dim)
    fit = Image.fromarray((a * 255).astype(np.uint8))
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=34, fill=255)
    im.paste(fit, (x0, y0), m)
    ImageDraw.Draw(im).rounded_rectangle(box, radius=34, outline=BORDER, width=3)


def reveal(i, bg, xray):
    """"One image made an invisible world visible": the 1895 radiograph lies hidden in fog, then on "visible" a quick
    scan reveals it sharp."""
    b = CFG["beats"]
    im = glitch_text(bg.copy(), ["22 December 1895"], i, b["reveal_title"], 150, px=76)
    box = (200, 320, 880, 1150)
    if xray is None:
        return im
    if i < b["shown"]:
        k = 0.0 if i >= b["hidden"] else 0.35
        _card(im, xray, box, sharp=k, dim=0.35 + 0.4 * k)
    else:
        g = min(1.0, (i - b["shown"]) / 6)                                      # a 6-frame reveal sweep
        fog = im.copy()
        _card(fog, xray, box, sharp=0.0, dim=0.35)
        _card(im, xray, box, sharp=1.0, dim=1.0)
        cut = int(box[1] + (box[3] - box[1]) * g)
        im.paste(fog.crop((0, cut, W, box[3] + 4)), (0, cut))
        if g < 1:
            ImageDraw.Draw(im).rectangle((box[0], cut - 3, box[2], cut + 3), fill=(150, 215, 255))
    return im


def show(i, bg):
    """"Don't just explain it. Show it.": "Explain it" is struck through and becomes "Show it"."""
    b = CFG["beats"]
    im = glitch_text(bg.copy(), ["Visualization"], i, b["show_title"], 150, px=86)
    d = ImageDraw.Draw(im)
    y = 520
    if i >= b["row"]:
        struck = i >= b["row_to"]
        d.text((W / 2, y), "Explain it", font=font(int(92 * pop(i - b["row"])), 760), fill=(118, 124, 145) if struck else INK, anchor="mm")
        if struck:
            g = min(1.0, (i - b["row_to"]) / 5)
            x0, _, x1, _ = d.textbbox((W / 2, y), "Explain it", font=font(92, 760), anchor="mm")
            d.line((x0 - 10, y + 2, x0 - 10 + (x1 - x0 + 20) * g, y + 2), fill=BLUE, width=8)
            d.line((W / 2, y + 66, W / 2, y + 108), fill=(118, 124, 145), width=5)
            d.polygon([(W / 2 - 14, y + 104), (W / 2 + 14, y + 104), (W / 2, y + 122)], fill=(118, 124, 145))
            d.text((W / 2, y + 190), "Show it", font=font(int(118 * pop(i - b["row_to"])), 840), fill=BLUE, anchor="mm")
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
    xray = Image.open(CFG["canvases"][0]["xray"]).convert("RGB") if CFG["canvases"][0].get("xray") else None
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
                c_im = reveal(i, bg, xray) if cv["kind"] == "reveal" else show(i, bg)
                m = masks.get((seg["clip"], sf))
                pic = np.asarray(panel(c_im, (head * 255).astype(np.uint8), None if m is None else np.asarray(m), head_top)).astype(np.float32) / 255
                canvas_on = True
            else:
                v = window(CFG["views"], t)["view"]
                frame = views[(seg["clip"], v)].get(sf)
                pic = grade_talking(frame.astype(np.float32) / 255, era, rng)
                if CFG["scan"][0] - 1e-6 <= t < CFG["scan"][1] - 1e-6:
                    pic = scan_fx(pic, (t - CFG["scan"][0]) / (CFG["scan"][1] - CFG["scan"][0]))
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
    json.dump([{"text": c["text"], "frame": f, "class": c["cls"]} for c, f in zip(caps, frames)], open(f"{E}/out/v17_captions.json", "w"), indent=1)
    print(f"rendered {N} frames ({N / FPS:.2f} s) -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{E}/out/v17_noaudio.mp4")
