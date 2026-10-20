#!/usr/bin/env python3
"""Reel cover in the client's "story" cover style (reference: the Brooklyn Bridge cover, "How did they / gain trust").

    python cover.py FRAME.png OUT_DIR --white "Would they" --red "believe you?" [--zoom 1.0] [--head-top 0.07]

Client rules: use a moment where the client is explaining or has a neutral face (eyes open, looking at the camera,
no blink); keep their own framing (no zoom on the head); keep the face natural (no coloured or heavily graded skin).
Look: the speaker is cut out (rembg u2net_human_seg, edges decontaminated) and kept nearly as filmed (only the
window's haze lifted, gentle contrast); the room becomes a dark, blurred navy backdrop; a vignette and a dark fade
under the chin carry two centred lines of heavy geometric sans (Montserrat ~760): the setup in white, the payoff
in red (#EA1B23). Layout (measured on the reference, 3:4): white line from 61% of the height, red line ending at
83%, the wider line ~87% of the width.
Writes OUT_DIR/cover_9x16.png (1080x1920, the reel cover; the 3:4 grid crop is its centre) and
OUT_DIR/cover_3x4.png (1080x1440, what the profile grid shows). Run it with the pipeline's python (rembg).
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(SKILL, "assets", "fonts", "cover", "Montserrat[wght].ttf")
RED, WHITE = (234, 27, 35), (255, 255, 255)
OW, OH = 1080, 1440                       # the 3:4 composition


def subject_mask(im):
    from rembg import new_session, remove
    small = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    m = remove(small, session=new_session("u2net_human_seg"), only_mask=True)
    m = m.resize(im.size, Image.BILINEAR).filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(3))   # choke + soft edge: no halo, no sticker look
    return np.asarray(m).astype(np.float32) / 255


def head_box(mask):
    """Top of the head, chin (approx.) and face centre from the subject mask."""
    h, w = mask.shape
    rows = np.where((mask > 0.5).any(axis=1))[0]
    top = rows[0]
    widths = (mask > 0.5).sum(axis=1)
    # the neck is the narrowest row between the head and the shoulders
    band = widths[top + int(0.05 * h): top + int(0.30 * h)]
    neck = top + int(0.05 * h) + int(np.argmin(band))
    cols = np.where(mask[top:neck].max(axis=0) > 0.5)[0]
    return top, neck, (cols[0] + cols[-1]) / 2


def _blur(x, r):
    im = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
    k = 4
    small = im.resize((im.width // k, im.height // k), Image.BILINEAR).filter(ImageFilter.GaussianBlur(r / k))
    return np.asarray(small.resize(im.size, Image.BILINEAR)).astype(np.float32) / 255


def relight(x, mask, face_rows):
    """The speaker stays natural (client: no coloured face, no heavy grade): only the haze of the window behind is
    taken off (black point), a gentle contrast curve, and nearly untouched colour. The mood comes from the backdrop,
    the vignette and the fade, not from the face."""
    luma = np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum = x @ luma
    sub = mask > 0.5
    black = np.percentile(lum[sub], 1.0) * 0.85
    x = np.clip((x - black) / (1 - black), 0, 1)                            # lift the veil only
    x = x * x * (3 - 2 * x) * 0.22 + x * 0.78                               # gentle contrast
    lum = (x @ luma)[..., None]
    x = lum + (x - lum) * 0.96                                              # colour almost as filmed
    return np.clip(x, 0, 1)


def backdrop(x):
    """The room, blurred and darkened to a navy night tone; bright areas become soft warm glows."""
    im = Image.fromarray((x * 255).astype(np.uint8))
    small = im.resize((im.width // 8, im.height // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(9))
    b = np.asarray(small.resize(im.size, Image.BILINEAR)).astype(np.float32) / 255
    lum = b @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    base = np.array([0.045, 0.062, 0.085], np.float32)                       # dark navy
    glow = np.clip((lum - 0.55) / 0.45, 0, 1)[..., None] ** 2
    out = base * (0.6 + 0.8 * lum[..., None]) + glow * np.array([0.30, 0.20, 0.12], np.float32)
    return np.clip(out, 0, 1)


def text_block(canvas, white, red, weight, top_frac=0.606, bottom_frac=0.83, width_frac=0.87):
    """Two centred lines in the reference's proportions, with a faint soft shadow."""
    W, H = canvas.size
    y0, y1 = top_frac * H, bottom_frac * H

    def font(px):
        f = ImageFont.truetype(FONT, px)
        f.set_variation_by_axes([weight])
        return f
    # size: the wider line spans width_frac of the frame, capped by the vertical band
    px = 200
    while px > 40:
        f = font(px)
        wmax = max(f.getbbox(t)[2] - f.getbbox(t)[0] for t in (white, red))
        asc, desc = f.getmetrics()
        if wmax <= width_frac * W and 2 * (asc + desc) * 0.92 <= (y1 - y0) * 1.12:
            break
        px -= 2
    f = font(px)
    asc, desc = f.getmetrics()
    line = (asc + desc) * 0.86
    base1 = y0 + asc * 0.92
    base2 = base1 + line
    shadow = Image.new("L", canvas.size, 0)
    sd = ImageDraw.Draw(shadow)
    for t, b in ((white, base1), (red, base2)):
        sd.text((W / 2, b + 4), t, font=f, fill=150, anchor="ms")
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    canvas = Image.composite(Image.new("RGB", canvas.size, (0, 0, 0)), canvas, shadow)
    d = ImageDraw.Draw(canvas)
    d.text((W / 2, base1), white, font=f, fill=WHITE, anchor="ms")
    d.text((W / 2, base2), red, font=f, fill=RED, anchor="ms")
    return canvas, px


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame")
    ap.add_argument("out")
    ap.add_argument("--white", required=True)
    ap.add_argument("--red", required=True)
    ap.add_argument("--weight", type=int, default=760)
    ap.add_argument("--zoom", type=float, default=1.0, help="1.0 = the client's own framing (full frame width); client: no zoom on the head")
    ap.add_argument("--head-top", type=float, default=0.07, help="where the top of the hair sits, fraction of the 3:4 height")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    src = Image.open(args.frame).convert("RGB")
    x = np.asarray(src).astype(np.float32) / 255
    mask = subject_mask(src)
    top, chin, cx = head_box(mask)
    m = mask ** 2.0           # tight edge: the hair's edge pixels carry the (dehazed, dark) ceiling
    rows = np.arange(x.shape[0], dtype=np.float32)[:, None, None]
    mouth = top + 0.78 * (chin - top)
    neck = 1 - 0.22 * np.clip((rows - mouth) / (0.45 * (chin - top)), 0, 1)   # the neck sits in the window's flare
    hh, ww = x.shape[:2]
    gy, gx = np.mgrid[:hh, :ww].astype(np.float32)
    glow = np.exp(-(((gx - cx) / (0.55 * ww)) ** 2 + ((gy - top - 0.35 * (chin - top)) / (0.30 * hh)) ** 2))
    back = backdrop(x) * 0.75 + glow[..., None] * np.array([0.022, 0.034, 0.048], np.float32)   # faint light on the wall behind
    subj = relight(x, mask, (top, chin)) * neck
    # edge decontamination: edge pixels mix hair with the bright room; refill them with colour spread outward from
    # the solid inside of the subject (normalised blur), so no light or dark outline remains
    core = (mask > 0.97).astype(np.float32)[..., None]
    fill = _blur(subj * core, 24) / np.maximum(_blur(np.repeat(core, 3, 2), 24), 1e-3)
    w_edge = np.clip((0.97 - mask) / 0.6, 0, 1)[..., None]
    subj = subj * (1 - w_edge) + fill * w_edge
    comp = subj * m[..., None] + back * (1 - m[..., None])
    # 3:4 window at the client's own framing (full width / zoom), the hair top at head_top; extended to 9:16 with
    # the backdrop above and the dark fade below
    crop_w = x.shape[1] / args.zoom
    crop_h = crop_w * OH / OW
    y_top = top - args.head_top * crop_h
    x_left = (x.shape[1] - crop_w) / 2 if args.zoom == 1.0 else cx - crop_w / 2
    full_h = crop_h * 1920 / 1440
    fy0 = y_top - (full_h - crop_h) / 2
    pad_b = int(max(0, fy0 + full_h - comp.shape[0])) + 2
    pad_t = int(max(0, -fy0)) + 2
    comp = np.concatenate([np.repeat(comp[:1], pad_t, 0), comp, np.repeat(comp[-1:], pad_b, 0)], axis=0)
    img = Image.fromarray((comp * 255 + 0.5).astype(np.uint8))
    box = (x_left, fy0 + pad_t, x_left + crop_w, fy0 + pad_t + full_h)
    cover = np.asarray(img.resize((1080, 1920), Image.LANCZOS, box=box)).astype(np.float32) / 255
    # vignette and the dark fade under the text (bottom of the 3:4 area and below)
    yy, xx = np.mgrid[:1920, :1080].astype(np.float32)
    u, v = (xx - 540) / 540, (yy - 960) / 960
    cover *= (1 - 0.32 * (u ** 2 * 0.8 + v ** 2 * 0.35))[..., None]
    chin_y = 240 + (args.head_top + (chin - top) / crop_h) * 1440             # chin in the 9:16 frame
    fade = np.clip((yy - (chin_y - 0.04 * 1440)) / (0.22 * 1440), 0, 1) ** 1.4
    cover *= (1 - 0.93 * fade)[..., None]                                       # clothes sink into black under the text
    rng = np.random.default_rng(1)
    cover += rng.normal(0, 0.014, cover.shape[:2]).astype(np.float32)[..., None]   # fine grain (hides the phone's smoothing)
    canvas = Image.fromarray((np.clip(cover, 0, 1) * 255 + 0.5).astype(np.uint8))
    canvas = canvas.filter(ImageFilter.UnsharpMask(radius=1.5, percent=35, threshold=3))
    grid = canvas.crop((0, 240, 1080, 1680))
    grid, px = text_block(grid, args.white, args.red, args.weight)
    canvas.paste(grid, (0, 240))
    canvas.save(os.path.join(args.out, "cover_9x16.png"))
    grid.save(os.path.join(args.out, "cover_3x4.png"))
    print(f"cover: head {top}-{chin}px, text {px}px -> {args.out}/cover_9x16.png, cover_3x4.png")


if __name__ == "__main__":
    main()
