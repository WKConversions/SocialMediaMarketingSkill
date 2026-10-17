#!/usr/bin/env python3
"""Contact sheets.
  sheet.py video <in.mp4> <out.png> [n=24] [start] [end] [cols=6]
      -> grid of n frames evenly spaced between start/end, each labelled with timestamp
  sheet.py vertical <out.png> <label=clip.mp4> [<label=clip.mp4> ...]
      -> for each vertical clip: 4 frames (10/40/70/95%) in a row, label on the left
  sheet.py image <out.png> <label=img> [...]   -> thumbnails of stills in a grid
"""
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def dur(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def grab(path, t, w):
    with tempfile.TemporaryDirectory() as td:
        o = os.path.join(td, "f.png")
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                        "-vf", f"scale={w}:-2", "-y", o], check=True)
        return Image.open(o).convert("RGB")


def video(inp, out, n=24, start=None, end=None, cols=6):
    d = dur(inp)
    s = float(start) if start not in (None, "") else 0.0
    e = float(end) if end not in (None, "") else d - 0.1
    ts = [s + (e - s) * (i + 0.5) / n for i in range(n)]
    frames = [grab(inp, t, 360) for t in ts]
    w, h = frames[0].size
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    dr = ImageDraw.Draw(sheet)
    f = font(20)
    for i, (fr, t) in enumerate(zip(frames, ts)):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(fr, (x, y))
        dr.rectangle([x, y, x + 90, y + 26], fill="black")
        dr.text((x + 4, y + 2), f"{t:6.1f}s", fill="yellow", font=f)
    sheet.save(out)
    print(out, sheet.size)


def vertical(out, pairs):
    rows = []
    f = font(22)
    for label, path in pairs:
        d = dur(path)
        frs = [grab(path, d * p, 216) for p in (0.1, 0.4, 0.7, 0.95)]
        fw, fh = frs[0].size
        row = Image.new("RGB", (300 + 4 * (fw + 6), fh + 10), (20, 20, 20))
        dr = ImageDraw.Draw(row)
        y = 10
        for line in [label[i:i + 22] for i in range(0, len(label), 22)] + [f"{d:.1f}s"]:
            dr.text((8, y), line, fill="white", font=f)
            y += 28
        for i, fr in enumerate(frs):
            row.paste(fr, (300 + i * (fw + 6), 5))
        rows.append(row)
    W = max(r.size[0] for r in rows)
    H = sum(r.size[1] for r in rows)
    sheet = Image.new("RGB", (W, H), "black")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.size[1]
    sheet.save(out)
    print(out, sheet.size)


def images(out, pairs, cols=4, tw=480):
    thumbs = []
    f = font(18)
    for label, p in pairs:
        im = Image.open(p)
        im.thumbnail((tw, tw))
        im = im.convert("RGB")
        c = Image.new("RGB", (tw, tw + 30), (25, 25, 25))
        c.paste(im, ((tw - im.size[0]) // 2, 30 + (tw - im.size[1]) // 2))
        ImageDraw.Draw(c).text((6, 4), label[:44], fill="yellow", font=f)
        thumbs.append(c)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (tw + 30)), "black")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * (tw + 30)))
    sheet.save(out)
    print(out, sheet.size)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "video":
        a = sys.argv[2:] + [None] * 6
        video(a[0], a[1], int(a[2] or 24), a[3], a[4], int(a[5] or 6))
    elif m == "vertical":
        vertical(sys.argv[2], [x.split("=", 1) for x in sys.argv[3:]])
    elif m == "image":
        images(sys.argv[2], [x.split("=", 1) for x in sys.argv[3:]])
