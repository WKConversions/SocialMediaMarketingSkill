"""Eiffel Tower 1887 -> WKC story edit, the classy / jazz style of the Wright edit (clips 7105-7112, 30 fps), version 1.

Picture:
  speaker   the client's footage, wide or a gentle 1.07 medium (the framing changes on every jump cut), lightly
            sharpened, then the approved near-original grade (tools/grade_classy.py: a little contrast, slightly
            lower mids, only the strongest reds calmed, soft vignette, fine grain).
  archival  fact-checked period photographs, the patent drawing and early flight film (out/media.json), toned like
            a fine silver print (warm blacks, ivory highlights), slow camera move; wide images sit across the frame
            over a dark, blurred copy of themselves with feathered edges.
  smooth    every change is a cross-dissolve (0.6 s into/out of an interlude, 0.4 s under the voice, 0.5 s between
            two archival shots). No flashes, no negative frames, no zoom punches, no motion design.
  captions  one style everywhere: Playfair Display Italic in warm ivory with a soft shadow, one line, centred at
            y = CAPTION_Y (over the suit, below the face, above the platform UI); a gentle 4-frame fade and 6 px
            rise in; the first visible frame is one frame before the word. Place/date labels in the interludes use
            the same style and spot.
  end card  the script's closing line on a warm black, the WKC logo small beneath.
Usage: python3 build_v20.py [OUT.mp4] [--frames 0,300,...  (stills only, for checking)] [--range A,B  (frames A..B-1 only)]
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/edit-10/tools")
from grade_classy import grade_classy, sharpen  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
E = "/home/user/SocialMediaMarketingSkill/lab/edit-10"
FONT = "/home/user/SocialMediaMarketingSkill/social-video-editor/assets/fonts/classy/PlayfairDisplay-Italic[wght].ttf"
W, H, FPS = 1080, 1920, 30
CFG = json.load(open(f"{E}/out/edit_v20.json"))
N = int(round(CFG["duration"] * FPS))
rng = np.random.default_rng(19)
CAPTION_Y = 1412                 # centre of the caption line (0.735 of the height)
CAPTION_PX, CAPTION_WGHT = 68, 500
IVORY = np.array([0.957, 0.925, 0.867], np.float32)
WARM_BLACK = np.array([0.045, 0.038, 0.034], np.float32)
FADE_IN, RISE = 4, 6
CLIP_FRAMES = {}


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def font(px=CAPTION_PX, wght=CAPTION_WGHT):
    f = ImageFont.truetype(FONT, px, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([wght])
    return f


# ---------------------------------------------------------------- speaker
class Reader:
    """Streams frames forward from a video; restarts on a backward seek; keeps the last few frames."""

    def __init__(self, path):
        self.path, self.proc, self.idx, self.cache = path, None, -1, {}

    def _open(self):
        if self.proc:
            self.proc.kill()
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", self.path, "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709",
                                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.idx = -1

    def get(self, k):
        if k in self.cache:
            return self.cache[k]
        if self.proc is None or k < self.idx:
            self._open()
        last = self.cache.get(self.idx)
        while self.idx < k:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                return last if last is not None else self.cache[max(self.cache)]
            self.idx += 1
            last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.cache[self.idx] = last
            for old in [c for c in self.cache if c < self.idx - 4]:
                del self.cache[old]
        return last


def n_frames(clip):
    if clip not in CLIP_FRAMES:
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", f"{E}/views/{clip}_wide.mp4"], capture_output=True, text=True).stdout
        CLIP_FRAMES[clip] = int(out.strip())
    return CLIP_FRAMES[clip]


def speaker_source(t):
    """(clip, view, source frame) for video time t; before a block's first segment / after its last one the clip
    runs on (its own earlier / later frames, held at the clip's ends) so dissolves always have a moving picture."""
    segs = CFG["speaker"]
    for s in segs:
        if s["start"] - 1e-6 <= t < s["end"] - 1e-6:
            return s["clip"], s["view"], s["src_start"] + (t - s["start"])
    # outside every segment: nearest block edge
    best = min(segs, key=lambda s: min(abs(t - s["start"]), abs(t - s["end"])))
    if t < best["start"]:
        return best["clip"], best["view"], best["src_start"] - (best["start"] - t)
    return best["clip"], best["view"], best["src_end"] + (t - best["end"])


def speaker_frame(t, views):
    clip, view, src = speaker_source(t)
    k = int(np.clip(round(src * FPS), 0, n_frames(clip) - 1))
    fr = sharpen(views[(clip, view)].get(k))
    return grade_classy(fr.astype(np.float32) / 255, rng=rng)


# ---------------------------------------------------------------- archival
def tone(lum):
    """Fine silver print: levels from the image's own percentiles, gentle S-curve, warm blacks -> ivory highlights."""
    lo, hi = np.percentile(lum, [0.5, 99.6])
    x = np.clip((lum - lo) / max(1e-3, hi - lo), 0, 1)
    x = x * x * (3 - 2 * x) * 0.30 + x * 0.70
    x = x * 0.86                                                             # never a hard white: keeps the low-tone feel
    return x


def print_colour(x):
    """Map 0..1 tone to warm monochrome (H, W, 3)."""
    lo = WARM_BLACK
    hi = np.array([0.94, 0.90, 0.82], np.float32)
    mid = np.array([0.50, 0.44, 0.37], np.float32)
    x3 = x[..., None]
    a = np.clip(x3 * 2, 0, 1)
    b = np.clip(x3 * 2 - 1, 0, 1)
    return np.where(x3 < 0.5, lo + (mid - lo) * a, mid + (hi - mid) * b).astype(np.float32)


_vig = {}


def vignette():
    if not _vig:
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        u, v = (xx - W / 2) / (W / 2), (yy - H * 0.47) / (H / 2)
        _vig[0] = (1 - 0.22 * np.clip(u ** 2 * 0.7 + v ** 2 * 0.6, 0, 1.4))[..., None].astype(np.float32)
    return _vig[0]


def compose(lum, sh, scale_up=1.0):
    """A toned 9:16 composite (float lum 0..1 in, HxWx3 out at scale_up x the output size)."""
    ow, oh = int(W * scale_up), int(H * scale_up)
    ih, iw = lum.shape
    im = Image.fromarray((lum * 255).astype(np.uint8))
    mode = sh.get("mode", "fit")
    if mode == "fill":
        win_h = min(ih, iw * 16 / 9) * sh.get("zoom", 1.0)
        win_w = win_h * 9 / 16
        cx, cy = sh.get("center", [0.5, 0.5])
        cx = float(np.clip(cx * iw, win_w / 2, iw - win_w / 2))
        cy = float(np.clip(cy * ih, win_h / 2, ih - win_h / 2))
        out = im.resize((ow, oh), Image.LANCZOS, box=(cx - win_w / 2, cy - win_h / 2, cx + win_w / 2, cy + win_h / 2))
        return np.asarray(out).astype(np.float32) / 255
    # fit: across the width (zoom >= 1 crops the sides), feathered top/bottom, over a dark blurred copy
    z = sh.get("zoom", 1.0)
    fw = int(ow * z)
    fh = int(ih * fw / iw)
    fg = np.asarray(im.resize((fw, fh), Image.LANCZOS)).astype(np.float32) / 255
    k = max(ow / iw, oh / ih)
    small = im.resize((max(1, int(iw * k / 10)), max(1, int(ih * k / 10))), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
    bg = np.asarray(small.resize((int(iw * k) + 2, int(ih * k) + 2), Image.BILINEAR)).astype(np.float32) / 255
    by, bx = (bg.shape[0] - oh) // 2, (bg.shape[1] - ow) // 2
    canvas = bg[by:by + oh, bx:bx + ow] * 0.16                               # a near-black warm field: the print floats
    cx = sh.get("center", [0.5, 0.5])[0]
    x0 = int(np.clip(ow / 2 - cx * fw, ow - fw, 0)) if fw > ow else (ow - fw) // 2
    y0 = int(oh * sh.get("y", 0.45) - fh / 2)
    feather = int(130 * scale_up)
    a = np.ones(fh, np.float32)
    ramp = smooth(np.linspace(0, 1, feather))
    a[:feather] = ramp
    a[-feather:] = ramp[::-1]
    ys, ye = max(0, y0), min(oh, y0 + fh)
    xs, xe = max(0, x0), min(ow, x0 + fw)
    part = fg[ys - y0: ye - y0, xs - x0: xe - x0]
    aa = a[ys - y0: ye - y0][:, None]
    canvas[ys:ye, xs:xe] = canvas[ys:ye, xs:xe] * (1 - aa) + part * aa
    return canvas


def diptych(lums, scale_up=1.0, gap=4):
    """Two images stacked (each filling its half by a centred crop), a thin warm-black line between."""
    ow, oh = int(W * scale_up), int(H * scale_up)
    hh = (oh - gap) // 2
    canvas = np.full((oh, ow), 0.0, np.float32)
    for j, lum in enumerate(lums):
        ih, iw = lum.shape
        k = max(ow / iw, hh / ih)
        im = Image.fromarray((lum * 255).astype(np.uint8)).resize((int(iw * k + 0.999), int(ih * k + 0.999)), Image.LANCZOS)
        x0, y0 = (im.width - ow) // 2, (im.height - hh) // 2
        canvas[j * (hh + gap): j * (hh + gap) + hh] = np.asarray(im.crop((x0, y0, x0 + ow, y0 + hh))).astype(np.float32) / 255
    return canvas


def load_lum(path, crop=None):
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im).astype(np.float32)
        lum = a / max(1.0, a.max())
    else:
        a = np.asarray(im.convert("RGB")).astype(np.float32) / 255
        lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    if crop:                                                                    # normalised [x0, y0, x1, y1]
        h, w = lum.shape
        lum = lum[int(crop[1] * h): int(crop[3] * h), int(crop[0] * w): int(crop[2] * w)]
    if lum.shape[1] > 3000:
        k = 3000 / lum.shape[1]
        lum = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((3000, int(lum.shape[0] * k)), Image.LANCZOS)).astype(np.float32) / 255
    return lum


class Still:
    """A toned print with a slow push-in (or drift) toward its focus point."""

    def __init__(self, sh):
        self.sh = sh
        self.dur = sh["end"] - sh["start"]
        self.k = 1.08
        if sh.get("files"):                                                     # a diptych: two portraits stacked
            comp = diptych([tone(load_lum(f, c)) for f, c in zip(sh["files"], sh["crops"])], self.k)
        else:
            comp = compose(tone(load_lum(sh["file"], sh.get("crop"))), sh, self.k)
        comp = comp * sh.get("gain", 1.0)                                       # e.g. a white patent sheet sits lower
        self.comp = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8))
        self.focus = sh.get("focus", [0.5, sh.get("y", 0.45)])
        self.push = sh.get("push", 0.06)

    def frame(self, t):
        if self.sh.get("push_span"):                                            # one move across several shots (a montage
            a, b = self.sh["push_span"]                                         # of registered prints stays registered)
            p = np.clip((self.sh["start"] + t - a) / (b - a), 0, 1)
        else:
            p = np.clip((t + 0.3) / (self.dur + 0.6), 0, 1)                     # the move runs through the dissolves
        z = 1 + self.push * p
        ow, oh = self.comp.size
        cw, ch = W * self.k / z, H * self.k / z
        cx = float(np.clip(ow / 2 + (self.focus[0] - 0.5) * ow * (z - 1) / z * 2, cw / 2, ow - cw / 2))
        cy = float(np.clip(oh / 2 + (self.focus[1] - 0.5) * oh * (z - 1) / z * 2, ch / 2, oh - ch / 2))
        out = self.comp.resize((W, H), Image.BICUBIC, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))
        return print_colour(np.asarray(out).astype(np.float32) / 255)


class Film:
    """Early flight film, toned like the prints, placed by compose() (fit across the width by default)."""

    def __init__(self, sh):
        self.sh = sh
        pre = 0.35                                                               # starts a little early (dissolve in)
        self.t0 = sh["start"] - pre
        src = max(0.0, sh.get("src", 0.0) - pre)
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{src:.3f}", "-i", sh["file"], "-vf",
                                      f"fps={FPS},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                              sh["file"]], capture_output=True, text=True).stdout.strip().split(",")
        self.w, self.h = int(out[0]), int(out[1])
        self.n, self.last, self.levels = 0, None, None

    def frame(self, t):
        want = int(round((t + self.sh["start"] - self.t0) * FPS))
        while self.n <= want:
            buf = self.proc.stdout.read(self.w * self.h)
            if len(buf) < self.w * self.h:
                break
            self.last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w).astype(np.float32) / 255
            self.n += 1
        lum = self.last
        if self.sh.get("crop"):
            c = self.sh["crop"]
            lum = lum[int(c[1] * self.h): int(c[3] * self.h), int(c[0] * self.w): int(c[2] * self.w)]
        # damaged frames on the old print (big black blotches) would flash: hold the last clean frame instead
        # (and single white flash frames: a sudden jump in very bright pixels)
        good = getattr(self, "good", None)
        if good is not None and ((lum < 0.06).mean() > 0.12 or (lum > 0.93).mean() - (good > 0.93).mean() > 0.03):
            lum = good
        else:
            self.good = lum
        if self.levels is None:                                                  # fixed levels (no pumping)
            self.levels = np.percentile(lum, [0.5, 99.6])
        lo, hi = self.levels
        x = np.clip((lum - lo) / max(1e-3, hi - lo), 0, 1)
        x = (x * x * (3 - 2 * x) * 0.30 + x * 0.70) * 0.86
        return print_colour(compose(x, self.sh))


# ---------------------------------------------------------------- captions
_cap_cache = {}


def caption_layer(text, px=CAPTION_PX):
    """(alpha of the text, alpha of its soft shadow), cropped to the text box + margin."""
    key = (text, px)
    if key not in _cap_cache:
        f = font(px)
        l, t, r, b = f.getbbox(text)
        pad = 40
        w, h = r - l + 2 * pad, b - t + 2 * pad
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=255)
        sh = im.filter(ImageFilter.GaussianBlur(9))
        _cap_cache[key] = (np.asarray(im).astype(np.float32) / 255, np.asarray(sh).astype(np.float32) / 255, pad)
    return _cap_cache[key]


def draw_text(base, text, alpha, yc, px=CAPTION_PX, colour=IVORY):
    a, s, pad = caption_layer(text, px)
    h, w = a.shape
    x0, y0 = (W - w) // 2, int(yc - h / 2)
    ys, xs = slice(max(0, y0), min(H, y0 + h)), slice(max(0, x0), min(W, x0 + w))
    aa = a[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0][..., None] * alpha
    ss = s[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0][..., None] * alpha
    sy = slice(min(H, ys.start + 3), min(H, ys.stop + 3))                       # shadow 3 px lower
    reg = base[sy, xs]
    base[sy, xs] = reg * (1 - 0.62 * ss[: reg.shape[0]])
    reg = base[ys, xs]
    base[ys, xs] = reg * (1 - aa) + colour * aa
    return base


def text_band(base, strength):
    """A soft darkening behind the caption line on archival images (bright skies), feathered, never a box."""
    if strength <= 0:
        return base
    y = np.arange(H, dtype=np.float32)
    band = np.exp(-((y - CAPTION_Y) / 150.0) ** 2) * 0.38 * strength
    return base * (1 - band)[:, None, None]


# ---------------------------------------------------------------- shots & dissolves
def shot_alpha(k, t, shots):
    sh = shots[k]
    d = CFG["dissolve"]
    prev = shots[k - 1] if k > 0 else None
    nxt = shots[k + 1] if k + 1 < len(shots) else None
    touch_prev = prev is not None and abs(prev["end"] - sh["start"]) < 1e-3
    touch_next = nxt is not None and abs(nxt["start"] - sh["end"]) < 1e-3
    if sh["start"] < 1e-6:
        a_in = 1.0
    else:
        din = d["within"] if touch_prev else (d["interlude"] if sh["part"].startswith("I") else d["under_voice"])
        a_in = smooth((t - (sh["start"] - din / 2)) / din)
    if touch_next:                                                               # the next shot dissolves over this one
        dn = d["within"]
        a_out = 1.0 if t < sh["end"] + dn / 2 else 0.0
    else:
        dout = d["interlude"] if sh["part"].split("+")[-1].startswith("I") else d["under_voice"]
        a_out = smooth(((sh["end"] + dout / 2) - t) / dout)
    return float(a_in * a_out)


def end_card(t_in, logo):
    bg = np.broadcast_to(WARM_BLACK * 0.9, (H, W, 3)).copy()
    g = smooth((t_in - 0.35) / 0.5)
    if g > 0:
        lines = CFG["end_card"]["text"]
        rise = (1 - g) * RISE
        for j, ln in enumerate(lines):
            bg = draw_text(bg, ln, g, 870 + j * 96 + rise, px=76)
    gl = smooth((t_in - 0.75) / 0.6)
    if gl > 0 and logo is not None:
        lg = logo.copy()
        lg.thumbnail((300, 160))
        la = np.asarray(lg).astype(np.float32) / 255
        h, w = la.shape[:2]
        x0, y0 = (W - w) // 2, 1150
        a = la[..., 3:4] * gl * 0.92
        bg[y0:y0 + h, x0:x0 + w] = bg[y0:y0 + h, x0:x0 + w] * (1 - a) + la[..., :3] * a
    return bg


# ---------------------------------------------------------------- main
def render_frame(i, views, shots, players, logo, caps):
    t = i / FPS
    end_start = CFG["end_card"]["start"]
    alphas = [shot_alpha(k, t, shots) if sh["start"] - 0.6 <= t <= sh["end"] + 0.6 else 0.0 for k, sh in enumerate(shots)]
    top = 0.0
    for a in alphas:                                                             # opacity of the archival stack
        top = top + a * (1 - top)
    e_card = smooth((t - (end_start - 0.3)) / 0.6) if t > end_start - 0.3 else 0.0
    base = None
    if top < 0.999 and e_card < 0.999:
        base = speaker_frame(t, views)
    for k, a in enumerate(alphas):
        if a <= 0:
            continue
        fr = players[k].frame(t - shots[k]["start"])
        g = rng.normal(0, 0.010, (H // 2, W // 2)).astype(np.float32)
        fr = np.clip(fr * vignette() + np.repeat(np.repeat(g, 2, 0), 2, 1)[:, :, None], 0, 1)
        base = fr if base is None or a >= 0.999 else base * (1 - a) + fr * a
    if base is None:
        base = np.zeros((H, W, 3), np.float32)
    if e_card > 0:
        base = base * (1 - e_card) + end_card(t - end_start, logo) * e_card
    # captions and labels: one at a time, same font, same spot
    live = [c for c in caps if c["frame"] <= i < c["end_frame"]]
    if live:
        c = live[-1]
        k = i - c["frame"]
        a = smooth((k + 1) / FADE_IN)
        if c.get("fade_out"):
            a *= smooth((c["end_frame"] - i) / 8)
        base = text_band(base, top * a)
        base = draw_text(base, c["text"], a, CAPTION_Y + (1 - smooth((k + 1) / (FADE_IN + 2))) * RISE)
    if t < 0.2:                                                                  # opening: up from black in 6 frames
        base = base * smooth(t / 0.2 + 1 / 6)
    return np.clip(base, 0, 1)


def main(out_path, only=None, rng_range=None):
    shots = CFG["shots"]
    missing = [s["slot"] for s in shots if "file" not in s and "files" not in s]
    if only is None and any(s.get("placeholder") for s in shots) and "pending" not in os.path.basename(out_path):
        raise SystemExit("media not yet fact-checked in out/media.json: render to a *pending* file only")
    if missing:
        raise SystemExit(f"unfilled visual slots: {missing} (out/media.json)")
    views = {(s["clip"], v): Reader(f"{E}/views/{s['clip']}_{v}.mp4") for s in CFG["speaker"] for v in ("wide", "medium")}
    players = [Film(s) if s.get("kind") == "video" else Still(s) for s in shots]
    logo = Image.open(CFG["logo"]).convert("RGBA") if CFG.get("logo") else None
    caps = CFG["captions"] + CFG["labels"]
    caps = sorted(caps, key=lambda c: c["frame"])
    if only is not None:
        for i in only:
            Image.fromarray((render_frame(i, views, shots, players, logo, caps) * 255 + 0.5).astype(np.uint8)).save(f"{out_path}_{i:04d}.png")
            for k, s in enumerate(shots):                                         # films stream forward only: reopen
                if s.get("kind") == "video":
                    players[k] = Film(s)
        return
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-vf", "scale=out_color_matrix=bt709", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    a, b = rng_range or (0, N)
    for i in range(a, b):
        enc.stdin.write((render_frame(i, views, shots, players, logo, caps) * 255 + 0.5).astype(np.uint8).tobytes())
        if i % 90 == 0:
            print(f"  frame {i}/{N}", flush=True)
    enc.stdin.close()
    enc.wait()
    json.dump([{"text": c["text"], "frame": c["frame"], "label": bool(c.get("label"))} for c in caps],
              open(f"{E}/out/v20_captions.json", "w"), indent=1)
    print(f"rendered frames {a}-{b - 1} of {N} ({N / FPS:.2f} s) -> {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    only = None
    if "--frames" in args:
        j = args.index("--frames")
        only = [int(x) for x in args[j + 1].split(",")]
        args = args[:j] + args[j + 2:]
    rr = None
    if "--range" in args:                                                       # render a part (parallel renders, then concat)
        j = args.index("--range")
        rr = tuple(int(x) for x in args[j + 1].split(","))
        args = args[:j] + args[j + 2:]
    main(args[0] if args else f"{E}/out/v20_noaudio.mp4", only, rr)
