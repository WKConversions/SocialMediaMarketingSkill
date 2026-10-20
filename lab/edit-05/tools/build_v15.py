"""Montgolfier 1783 -> WKC story edit (clips c1-c4), version 1.

Picture per output frame comes from one of: the talking head (wide/medium/close of the clip that is speaking,
graded 'history' for the 1783 part and 'now' for the WKC part, tools/grade.py), a real 1783 print (slow camera
move, print treatment), or a canvas moment (outreach message / website showcase, speaker in the bottom panel
with the head popping out over its edge). Prime captions on top (centred, Difference blend; above the panel in
canvas mode). Pacing (uniform pauses) comes from out/edl.json, creative decisions from out/edit_v15.json.
Usage: python3 build_v15.py [OUT.mp4]
"""
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/edit-05/tools")
import quick_prime as qp  # noqa: E402
from grade import grade_print, grade_talking  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
LAB = "/home/user/SocialMediaMarketingSkill/lab"
E = f"{LAB}/edit-05"
W, H, FPS = 1080, 1920, 30
EDL = json.load(open(f"{E}/out/edl.json"))
CFG = json.load(open(f"{E}/out/edit_v15.json"))
SPEECH_END = EDL["duration"]
N = int(round((SPEECH_END + CFG["end_card_s"]) * FPS))
rng = np.random.default_rng(7)
PUNCH = 0.04                      # client: don't push far into the face
NOW_CLIPS = ("c3", "c4")          # the WKC part gets the 'now' grade

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
        out = self.im.resize((W, H), Image.BICUBIC, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))
        return np.asarray(out).astype(np.float32) / 255


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


def outreach(i, bg):
    b = CFG["beats"]
    im = glitch_text(bg.copy(), ["Your outreach"], i, b["outreach_title"], 150, px=86)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 110, 400, 970, 1160
    d.rounded_rectangle((x0, y0, x1, y1), radius=34, fill=CARD, outline=BORDER, width=3)
    # header: new message to your next client
    d.ellipse((x0 + 36, y0 + 34, x0 + 106, y0 + 104), fill=(46, 56, 86))
    d.text((x0 + 71, y0 + 69), "?", font=font(40, 800), fill=INK, anchor="mm")
    d.text((x0 + 130, y0 + 52), "Your next client", font=font(38, 760), fill=INK, anchor="lm")
    d.text((x0 + 130, y0 + 90), "New message", font=font(28, 500), fill=MUTED, anchor="lm")
    d.line((x0 + 30, y0 + 136, x1 - 30, y0 + 136), fill=(40, 48, 72), width=2)
    # the message bubble, typed in from "saying"
    if i >= b["bubble"]:
        text = "Hi! We can do this for you:"
        n = min(len(text), int((i - b["bubble"]) * 2.2) + 1)
        bx0, by0, bx1 = x0 + 50, y0 + 176, x1 - 50
        rows_on = [k for k, f in enumerate(b["rows"]) if i >= f]
        by1 = by0 + 110 + 104 * len(rows_on)
        d.rounded_rectangle((bx0, by0, bx1, by1), radius=30, fill=(30, 40, 70))
        d.text((bx0 + 40, by0 + 56), text[:n], font=font(42, 650), fill=INK, anchor="lm")
        newest = rows_on[-1] if rows_on else -1
        for k in rows_on:
            s = pop(i - b["rows"][k])
            ry = by0 + 150 + 104 * k
            r = 24 * s
            d.ellipse((bx0 + 64 - r, ry - r, bx0 + 64 + r, ry + r), fill=BLUE if k == newest else (38, 74, 122))
            d.line((bx0 + 53, ry, bx0 + 62, ry + 9, bx0 + 77, ry - 9), fill=(255, 255, 255), width=5)
            dim = 255 if k == newest else 175
            d.text((bx0 + 112, ry), CFG["services"][k], font=font(int(52 * s), 760), fill=(dim, dim, dim), anchor="lm")
    # send bar
    sy = y1 - 104
    d.rounded_rectangle((x0 + 40, sy, x1 - 40, sy + 70), radius=35, outline=(52, 62, 92), width=2)
    sent = i >= b["sent"]
    btn = BLUE if sent else (52, 62, 92)
    s = pop(i - b["sent"]) if sent else 1.0
    r = 30 * s
    d.ellipse((x1 - 105 - r, sy + 35 - r, x1 - 105 + r, sy + 35 + r), fill=btn)
    d.polygon([(x1 - 118, sy + 22), (x1 - 88, sy + 35), (x1 - 118, sy + 48)], fill=(255, 255, 255) if sent else INK)
    d.text((x0 + 80, sy + 35), "Sent" if sent else "Write a message...", font=font(32, 600), fill=BLUE if sent else MUTED, anchor="lm")
    return im


def showcase(i, bg, web):
    b = CFG["beats"]
    im = glitch_text(bg.copy(), ["Show, don't tell"], i, b["showcase_title"], 150, px=86)
    d = ImageDraw.Draw(im)
    pw, ph = 470, 836                                   # phone with the client's site scrolling
    px0, py0 = (W - pw) // 2, 352
    d.rounded_rectangle((px0 - 14, py0 - 14, px0 + pw + 14, py0 + ph + 14), radius=58, fill=(4, 6, 12), outline=BORDER, width=3)
    if web is not None:
        f0 = int(round(16.80 * FPS))
        shot = Image.fromarray(web.get(max(0, i - f0) + 24)).resize((pw, int(pw * H / W)), Image.LANCZOS).crop((0, 0, pw, ph))   # skip the 1 s hold
    else:
        shot = Image.new("RGB", (pw, ph), (240, 238, 232))
        ImageDraw.Draw(shot).text((pw / 2, ph / 2), "wkconversions.com", font=font(40, 760), fill=NAVY, anchor="mm")
    m = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, pw, ph), radius=46, fill=255)
    im.paste(shot, (px0, py0), m)
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
    views = {(clip, v): Reader(f"{E}/views/{clip}_{v}.mp4") for clip in ("c1", "c2", "c3", "c4") for v in ("wide", "medium", "close")}
    shots = {k: PrintShot(ins) for k, ins in enumerate(CFG["inserts"])}
    web = Reader(CFG["website"]) if CFG.get("website") else None
    logo = Image.open(CFG["logo"]).convert("RGBA") if CFG.get("logo") else None
    masks = {}
    import glob
    import os
    for p in glob.glob(f"{E}/masks/c3/m*.png"):
        masks[int(os.path.basename(p)[1:6])] = Image.open(p).convert("L")
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
                pic = grade_print(shots[k].frame(t - ins["start"]), rng, shots[k].colour)
            elif cv is not None:
                frame = views[(seg["clip"], "wide")].get(sf)
                head = grade_talking(frame.astype(np.float32) / 255, era, rng)
                c_im = outreach(i, bg) if cv["kind"] == "outreach" else showcase(i, bg, web)
                m = masks.get(sf)
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
    json.dump([{"text": c["text"], "frame": f, "class": c["cls"]} for c, f in zip(caps, frames)], open(f"{E}/out/v15_captions.json", "w"), indent=1)
    print(f"rendered {N} frames ({N / FPS:.2f} s) -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{E}/out/v15_noaudio.mp4")
