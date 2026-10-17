"""Burn the plan's captions into the clip in a caption style.

Typesetting (style["classes"], fonts from style["fonts"], sizes scale with the frame's short side):
  sans    a variable font at fixed axes, lowercase, glyphs placed one by one: advance from pair widths
          (kerning) plus tracking.
  script  lowercase in the script font, the capital in a swash font (scaled, slightly overlapped), cropped
          to its ink and scaled to a fixed ink height.
Everything is drawn supersampled, cropped to the ink and downsampled. Words wider than the layout's
max width are scaled down to fit. Each caption's ink box is centred on the layout point (Prime: the frame
centre, on every kind of footage).

Compositing per frame in float, display-referred RGB (like After Effects / CSS mix-blend-mode):
    difference: out = B * (1 - a) + |C - B| * a
so the letters show the negative of whatever moves behind them. A caption is visible from its first
frame (ceil(start * fps)) until the next caption's first frame. While a zoom caption is visible the
composite is scaled about the zoom origin: s(t) = 1 + amount * (1 - exp(-(t - start) / tau_s)).

Frames come from the source through common.Clip (trim, crop, scale, bt709 -> RGB) and go to x264 as
video only; the mix step adds the audio.
"""
import math
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import PipelineError, first_frame, hex_rgb, skill_path

AXIS_TAGS = {"optical size": "opsz", "weight": "wght", "width": "wdth", "slant": "slnt", "italic": "ital"}


class Typesetter:
    """Caption text -> anti-aliased alpha (float32, cropped to the ink) at the output size."""

    def __init__(self, style, scale):
        self.style = style
        self.scale = scale                       # output short side / style reference short side
        self.ss = int(style["render"]["supersample"])
        self.max_w = style["layout"]["max_width_px"] * scale

    def font(self, name, size):
        spec = self.style["fonts"][name]
        f = ImageFont.truetype(skill_path(spec["file"]), max(1, int(round(size))), layout_engine=ImageFont.Layout.RAQM)
        if spec.get("axes"):
            values = []
            for axis in f.get_variation_axes():
                tag = AXIS_TAGS.get(axis["name"].decode().lower() if isinstance(axis["name"], bytes) else axis["name"].lower())
                values.append(min(max(spec["axes"].get(tag, axis["default"]), axis["minimum"]), axis["maximum"]))
            f.set_variation_by_axes(values)
        return f

    def sans(self, text, cfg):
        px = cfg["size_px"] * self.scale * self.ss
        f = self.font(cfg["font"], px)
        track = cfg["tracking_em"] * px
        xs, x = [], 0.0
        for i, ch in enumerate(text):
            xs.append(x)
            nxt = text[i + 1] if i + 1 < len(text) else ""
            adv = f.getlength(ch + nxt) - f.getlength(nxt) if nxt else f.getlength(ch)
            x += adv + (track if ch != " " else 0)
        canvas = Image.new("L", (int(x + px), int(px * 2)), 0)
        d = ImageDraw.Draw(canvas)
        for ch, cx in zip(text, xs):
            d.text((cx + px * 0.2, int(px * 1.3)), ch, font=f, fill=255, anchor="ls")
        return canvas

    def script(self, text, cfg):
        size = int(cfg["draw_px"] * self.ss)
        body = self.font(cfg["font"], size)
        cap = self.font(cfg["capital_font"], size * cfg["capital_scale"])
        canvas = Image.new("L", (size * (len(text) + 2), size * 3), 0)
        d = ImageDraw.Draw(canvas)
        x, base = size * 0.5, size * 2
        rest = text
        if text[:1].isupper():
            d.text((x, base), text[0], font=cap, fill=255, anchor="ls")
            x += cap.getlength(text[0]) - size * cfg["capital_overlap_em"]
            rest = text[1:]
        d.text((x, base), rest, font=body, fill=255, anchor="ls")
        canvas = canvas.crop(canvas.getbbox())
        k = cfg["ink_height_px"] * self.scale * self.ss / canvas.height
        if canvas.width * k > self.max_w * self.ss:
            k = self.max_w * self.ss / canvas.width
        return canvas.resize((max(1, int(canvas.width * k)), max(1, int(canvas.height * k))), Image.LANCZOS)

    def alpha(self, text, cls):
        cfg = self.style["classes"][cls]
        img = self.script(text, cfg) if cls == "script" else self.sans(text, cfg)
        box = img.getbbox()
        if box is None:
            raise PipelineError(f"caption {text!r} renders no ink")
        img = img.crop(box)
        img = img.resize((max(1, img.width // self.ss), max(1, img.height // self.ss)), Image.LANCZOS)
        if img.width > self.max_w:
            k = self.max_w / img.width
            img = img.resize((int(self.max_w), max(1, int(img.height * k))), Image.LANCZOS)
        return np.asarray(img, np.float32) / 255.0


@dataclass
class Layer:
    """One caption placed on the frame."""
    text: str
    x0: int
    y0: int
    alpha: np.ndarray
    colour: np.ndarray
    first: int              # first visible frame
    last: int               # first frame of the next caption (exclusive)
    start: float            # planned start (s): phase of the zoom
    zoom: bool


def layers(plan, style, width, height):
    ts = Typesetter(style, min(width, height) / style["reference_short_side_px"])
    lay = style["layout"]
    if lay["placement"] != "centre":
        raise PipelineError(f"unsupported placement {lay['placement']!r}")
    fps, caps = plan["clip"]["fps"], plan["captions"]
    out = []
    for k, c in enumerate(caps):
        a = ts.alpha(c["text"], c["class"])
        h, w = a.shape
        out.append(Layer(text=c["text"], x0=int(lay["x"] * width - w / 2), y0=int(lay["y"] * height - h / 2), alpha=a,
                         colour=hex_rgb(style["colours"][c["fill"]]), first=first_frame(c["start"], fps),
                         last=first_frame(caps[k + 1]["start"], fps) if k + 1 < len(caps) else plan["clip"]["frames"],
                         start=c["start"], zoom=bool(c["zoom"])))
    return out


def composite(frame, layer):
    """Difference-blend one caption into an (H, W, 3) uint8 frame; returns a new frame."""
    H, W = frame.shape[:2]
    h, w = layer.alpha.shape
    ys = slice(max(0, layer.y0), min(H, layer.y0 + h))
    xs = slice(max(0, layer.x0), min(W, layer.x0 + w))
    a = layer.alpha[ys.start - layer.y0: ys.stop - layer.y0, xs.start - layer.x0: xs.stop - layer.x0][..., None]
    bg = frame[ys, xs].astype(np.float32) / 255.0
    out = frame.copy()
    out[ys, xs] = (np.clip(bg * (1 - a) + np.abs(layer.colour - bg) * a, 0, 1) * 255 + 0.5).astype(np.uint8)
    return out


def zoom_at(layer, t, zcfg):
    if not layer.zoom:
        return 1.0
    return 1.0 + zcfg["amount"] * (1.0 - math.exp(-(t - layer.start) / zcfg["tau_s"]))


def apply_zoom(frame, s, origin):
    """Scale an (H, W, 3) uint8 frame by s about origin (fractions of the frame)."""
    H, W = frame.shape[:2]
    cw, ch = W / s, H / s
    ox, oy = origin[0] * W, origin[1] * H
    img = Image.fromarray(frame).resize((W, H), Image.BICUBIC, box=(ox - cw / 2, oy - ch / 2, ox + cw / 2, oy + ch / 2))
    return np.asarray(img)


def visible(layers_, k):
    return next((ly for ly in layers_ if ly.first <= k < ly.last), None)


def finished_frame(bg, k, layers_, style, fps):
    """The frame k of the finished video: the visible caption composited, then the zoom."""
    ly = visible(layers_, k)
    if ly is None:
        return bg
    out = composite(bg, ly)
    s = zoom_at(ly, k / fps, style["zoom"])
    if s > 1.0 + style["zoom"]["min_visible"]:
        out = apply_zoom(out, s, style["zoom"]["origin"])
    return out


def render(clip, plan, style, out_path, scaler):
    """Write the captioned clip (video only) to out_path."""
    lays = layers(plan, style, clip.width, clip.height)
    r = style["render"]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{clip.width}x{clip.height}", "-r", str(clip.fps), "-i", "-", "-an",
           "-vf", f"scale=out_color_matrix={r['colour']}:out_range=tv,format={r['pix_fmt']}",
           "-c:v", r["video_codec"], "-crf", str(r["crf"]), "-preset", r["preset"], "-pix_fmt", r["pix_fmt"],
           "-colorspace", r["colour"], "-color_primaries", r["colour"], "-color_trc", r["colour"], "-color_range", "tv",
           "-movflags", "+faststart", out_path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        for k, bg in enumerate(clip.frames(scaler)):
            enc.stdin.write(finished_frame(bg, k, lays, style, clip.fps).tobytes())
            n += 1
    finally:
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        enc.wait()
    if enc.returncode != 0:
        raise PipelineError(f"x264 encode failed:\n{err[-800:]}")
    return dict(frames=n, captions=[dict(text=ly.text, first_frame=ly.first, last_frame=ly.last - 1,
                                         size=list(ly.alpha.shape[::-1]), x0=ly.x0, y0=ly.y0) for ly in lays])
