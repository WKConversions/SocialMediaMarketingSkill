"""Quick Prime-style caption render for the client's talking-head test clip.

A fast first pass while the full data-driven renderer is being built:
white (or pink) text in Difference blend, Inter for body words, Pinyon Script
with an Imperial Script capital for emphasis words, hard cuts between words,
a ~12% punch-in zoom on every script word, and one script word behind the
speaker's head using the subject masks.

Usage: python3 quick_prime.py CLIP.mp4 MASK_DIR OUT.mp4
"""
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB = "/home/user/SocialMediaMarketingSkill/lab/caption-style-01"
INTER = f"{LAB}/fonts/inter/Inter-Variable.ttf"
PINYON = f"{LAB}/fonts/pinyonscript/PinyonScript-Regular.ttf"
IMPERIAL = f"{LAB}/fonts/imperialscript/ImperialScript-Regular.ttf"

W, H, FPS = 1080, 1920, 30
SS = 2                       # supersampling for clean glyph edges
WHITE = (1.0, 1.0, 1.0)
PINK = (226 / 255, 139 / 255, 139 / 255)   # the reference's "Prime" fill

# (text, start s, end s, class, fill, y_centre px, behind_subject)
LOW = 1500                   # chin / lit jacket: Difference has something to react to
PLAN = [
    ("dit is een", 0.14, 0.94, "sans", WHITE, LOW, False),
    ("Testvideo", 0.94, 1.49, "script", PINK, 670, True),
    ("om te", 1.49, 1.80, "sans", WHITE, LOW, False),
    ("kijken", 1.80, 2.16, "sans", WHITE, LOW, False),
    ("of jij", 2.16, 2.43, "sans", WHITE, LOW, False),
    ("goed", 2.43, 2.73, "sans", WHITE, LOW, False),
    ("kan", 2.73, 2.93, "sans", WHITE, LOW, False),
    ("Editen", 2.93, 3.34, "script", WHITE, LOW - 20, False),
    ("wat ik", 3.34, 3.98, "sans", WHITE, LOW, False),
    ("graag zou", 3.98, 4.27, "sans", WHITE, LOW, False),
    ("Willen", 4.27, 9.99, "script", WHITE, LOW - 20, False),
]
SANS_PX = 112
SANS_TRACK_EM = -0.07
SCRIPT_INK_H = 270           # target ink height of a capitalised script word
MAX_W = 960
ZOOM_AMOUNT, ZOOM_TAU = 0.12, 4.0   # punch-in on script words, exponential ease (frames)


def inter(px):
    f = ImageFont.truetype(INTER, px, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([32, 780])          # axes: opsz, wght
    return f


def render_sans(text, px):
    f = inter(px * SS)
    track = SANS_TRACK_EM * px * SS
    # per-glyph placement with kerning from pair widths, plus tracking
    xs, x = [], 0.0
    for i, ch in enumerate(text):
        xs.append(x)
        nxt = text[i + 1] if i + 1 < len(text) else ""
        adv = f.getlength(ch + nxt) - f.getlength(nxt) if nxt else f.getlength(ch)
        x += adv + (track if ch != " " else 0)
    canvas = Image.new("L", (int(x + px * SS), int(px * SS * 2)), 0)
    d = ImageDraw.Draw(canvas)
    base = int(px * SS * 1.3)
    for ch, cx in zip(text, xs):
        d.text((cx + px * SS * 0.2, base), ch, font=f, fill=255, anchor="ls")
    return canvas


def render_script(text, ink_h):
    size = 200 * SS
    pin = ImageFont.truetype(PINYON, size, layout_engine=ImageFont.Layout.RAQM)
    imp = ImageFont.truetype(IMPERIAL, int(size * 1.08), layout_engine=ImageFont.Layout.RAQM)
    canvas = Image.new("L", (size * (len(text) + 2), size * 3), 0)
    d = ImageDraw.Draw(canvas)
    x, base = size * 0.5, size * 2
    if text[0].isupper():
        d.text((x, base), text[0], font=imp, fill=255, anchor="ls")
        x += imp.getlength(text[0]) - size * 0.06
        rest = text[1:]
    else:
        rest = text
    d.text((x, base), rest, font=pin, fill=255, anchor="ls")
    bbox = canvas.getbbox()
    canvas = canvas.crop(bbox)
    scale = ink_h * SS / canvas.height
    if canvas.width * scale > MAX_W * SS:
        scale = MAX_W * SS / canvas.width
    return canvas.resize((max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))), Image.LANCZOS)


def word_alpha(text, cls):
    img = render_sans(text, SANS_PX) if cls == "sans" else render_script(text, SCRIPT_INK_H)
    img = img.crop(img.getbbox())
    img = img.resize((max(1, img.width // SS), max(1, img.height // SS)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def main(clip, mask_dir, out):
    alphas = [word_alpha(text, cls) for text, _s, _e, cls, *_rest in PLAN]
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip, "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-i", clip, "-map", "0:v", "-map", "1:a", "-vf", "scale=out_color_matrix=bt709",
                            "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
                            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-c:a", "copy", "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = 0
    while True:
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        n += 1
        t = (n - 1) / FPS
        B = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
        out_f = B.copy()
        zoom = 1.0
        for (text, s, e, cls, fill, yc, behind), a in zip(PLAN, alphas):
            if not (s <= t < e):
                continue
            h, w = a.shape
            x0, y0 = (W - w) // 2, int(yc - h / 2)
            ys, xs_ = slice(max(0, y0), min(H, y0 + h)), slice(max(0, x0), min(W, x0 + w))
            aa = a[ys.start - y0: ys.stop - y0, xs_.start - x0: xs_.stop - x0]
            if behind:
                m = np.asarray(Image.open(f"{mask_dir}/m{n:05d}.png"), np.float32)[ys, xs_] / 255.0
                aa = aa * (1.0 - m)
            aa = aa[..., None]
            bg = out_f[ys, xs_]
            out_f[ys, xs_] = bg * (1 - aa) + np.abs(np.array(fill, np.float32) - bg) * aa
            if cls == "script":
                k = (t - s) * FPS
                zoom = 1.0 + ZOOM_AMOUNT * (1.0 - np.exp(-k / ZOOM_TAU))
        frame = Image.fromarray((np.clip(out_f, 0, 1) * 255 + 0.5).astype(np.uint8))
        if zoom > 1.0005:
            cw, ch = W / zoom, H / zoom
            frame = frame.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"rendered {n} frames -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
