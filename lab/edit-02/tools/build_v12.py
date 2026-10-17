"""Version 12 of the client's test clip: Prime captions plus what the client pointed out in three references.

- Quick view switching (reel 1): wide / medium / close views cut from the 4K source; the first 1-2
  frames of a new view are shown in negative; a short switching burst with in-shot flicker.
- Canvas mode (reel 2): the speaker shrinks into a rounded panel at the bottom with the head cut out
  over the panel edge; above, a visual of what is being said (title + an edit timeline whose clips
  pop in).
- Sound per moment (reel 3) is written as an event list for sfx_v12.py: flash on quick cuts, whoosh
  into/out of the canvas, glitch on the title, clicks on pop-ups, riser into the last word; captions
  keep the client's shutter.

Usage: python3 build_v12.py OUT_VIDEO_NOAUDIO.mp4 EVENTS.json
"""
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
import quick_prime as qp  # caption renderer (fonts, Difference blend)

LAB = "/home/user/SocialMediaMarketingSkill/lab"
VIEWS = {v: f"{LAB}/edit-02/views/{v}.mp4" for v in ("wide", "medium", "close")}
MASKS = f"{LAB}/caption-style-01/footage/user/masks"      # m00001 = clip.mp4 frame 0
MASK_OFFSET = 14                                          # clip2 frame i  ->  m{i + 14}
PLAN = f"{LAB}/caption-style-01/out/plan_v11.json"
W, H, FPS, N = 1080, 1920, 30, 131
WHITE, PINK = qp.WHITE, qp.PINK

# ---- edit decision list (frame numbers, 0-based) -------------------------------------------------
# view per frame range; canvas mode replaces the picture with the canvas layout
SEGMENTS = [  # (first frame, view)
    (0, "wide"),       # dit is een
    (15, "close"),     # Testvideo            (quick cut, negative flash)
    (35, "canvas"),    # om te kijken of jij goed kan   (whoosh in)
    (77, "close"),     # Editen               (whoosh out)
    (99, "wide"),      # wat ik    -- burst --
    (106, "close"),    # graag
    (114, "medium"),   # zou
    (119, "close"),    # Willen               (riser lands here)
]
NEGATIVE = {15, 16, 35, 77, 99, 106, 107, 114, 122, 125}   # 1-2 negative frames on switches + flicker
CANVAS_IN, CANVAS_OUT = 35, 77
TITLE_AT = 43                       # "kijken": the canvas title appears with a glitch
POPS = [39, 47, 56, 67]             # timeline clips pop in (between caption frames, so clicks don't collide)
CAPTION_Y_CANVAS = 1235

# ---- canvas look (after the reference: dark, dot grid, white bold title, card with thin border) -
BG = (10, 10, 11)
DOT = (30, 30, 33)
CARD = (18, 18, 20)
BORDER = (232, 232, 232)
RED = (229, 57, 53)
BLOCK_COLOURS = [(42, 120, 214), (235, 104, 52), (210, 210, 205), (42, 120, 214), (235, 104, 52)]
PANEL = dict(x0=135, x1=945, top=1390, radius=46, scale=0.75, head_top=1310)


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


def segment_start(i):
    return max(f0 for f0, _ in SEGMENTS if f0 <= i)


def read_frames(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def canvas_background():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for y in range(18, H, 36):
        for x in range(18, W, 36):
            d.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=DOT)
    return im


TRACKS = [  # (label, y, blocks as (x0, x1) fractions of the track width, pop index)
    ("V2", 0, [(0.10, 0.28, 1), (0.55, 0.70, 3)]),
    ("V1", 1, [(0.00, 0.22, 0), (0.24, 0.47, 1), (0.49, 0.74, 2), (0.76, 1.00, 3)]),
    ("A1", 2, [(0.00, 0.47, 0), (0.49, 1.00, 2)]),
]


def draw_canvas(i, bg):
    im = bg.copy()
    d = ImageDraw.Draw(im)
    k = i - CANVAS_IN
    # title (glitches in on TITLE_AT: 3 frames of channel offsets / slice shifts, then clean)
    if i >= TITLE_AT:
        tf = font(84)
        layer = Image.new("L", (W, 260), 0)
        ld = ImageDraw.Draw(layer)
        for li, line in enumerate(["Kan jij goed", "editen?"]):
            ld.text((W / 2, 70 + li * 100), line, font=tf, fill=255, anchor="mm")
        g = i - TITLE_AT
        arr = np.asarray(layer).astype(np.float32) / 255
        rgb = np.zeros((260, W, 3), np.float32)
        if g < 3:
            off = [14, 0, -14] if g != 1 else [-9, 5, 11]
            for c in range(3):
                rgb[..., c] = np.roll(arr, off[c], axis=1)
            for s in range(0, 260, 26):                      # horizontal slice displacement
                if (s // 26 + g) % 3 == 0:
                    rgb[s:s + 26] = np.roll(rgb[s:s + 26], 24 * (1 if g % 2 else -1), axis=1)
        else:
            rgb[:] = arr[..., None]
        base = np.asarray(im).astype(np.float32) / 255
        region = base[200:460]
        base[200:460] = np.maximum(region, rgb)
        im = Image.fromarray((base * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
    # card with an edit timeline
    cx0, cy0, cx1, cy1 = 120, 500, 960, 1100
    d.rounded_rectangle((cx0, cy0, cx1, cy1), radius=30, fill=CARD, outline=BORDER, width=3)
    lf = font(30, 600)
    d.text((cx0 + 40, cy0 + 42), "EDIT", font=lf, fill=(150, 150, 150), anchor="lm")
    tx0, tx1 = cx0 + 120, cx1 - 40
    popped = sum(1 for p in POPS if i >= p)
    for label, row, blocks in TRACKS:
        ty = cy0 + 130 + row * 130
        d.text((cx0 + 40, ty + 40), label, font=lf, fill=(120, 120, 120), anchor="lm")
        d.line((tx0, ty + 40, tx1, ty + 40), fill=(40, 40, 44), width=2)
        for bi, (a, b, pop) in enumerate(blocks):
            if pop >= popped:
                continue
            age = i - POPS[pop]
            s = 0.6 + 0.4 * min(1, age / 2)                 # quick 2-frame pop-in
            bx0, bx1 = tx0 + a * (tx1 - tx0) + 4, tx0 + b * (tx1 - tx0) - 4
            mx, my = (bx0 + bx1) / 2, ty + 40
            hw, hh = (bx1 - bx0) / 2 * s, 38 * s
            d.rounded_rectangle((mx - hw, my - hh, mx + hw, my + hh), radius=12, fill=BLOCK_COLOURS[(row + bi) % len(BLOCK_COLOURS)])
    # playhead sweeping across while he talks, with a red dot like the reference timelines
    p = min(1, k / (CANVAS_OUT - CANVAS_IN - 1))
    px = tx0 + p * (tx1 - tx0)
    d.line((px, cy0 + 100, px, cy1 - 60), fill=RED, width=4)
    d.ellipse((px - 13, cy0 + 87, px + 13, cy0 + 113), fill=RED)
    return im


def panel_composite(canvas_im, frame_rgb, mask_l):
    """Speaker in a rounded panel at the bottom; the part of the subject above the panel edge pops out."""
    s = PANEL["scale"]
    fw, fh = int(W * s), int(H * s)
    small = Image.fromarray(frame_rgb).resize((fw, fh), Image.LANCZOS)
    msmall = Image.fromarray(mask_l).resize((fw, fh), Image.BILINEAR)
    y_off = int(PANEL["head_top"] - 700 * s)
    x_off = (W - fw) // 2
    layer = Image.new("RGB", (W, H))
    layer.paste(small, (x_off, y_off))
    mlayer = Image.new("L", (W, H), 0)
    mlayer.paste(msmall, (x_off, y_off))
    panel_mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(panel_mask).rounded_rectangle((PANEL["x0"], PANEL["top"], PANEL["x1"], H + PANEL["radius"]),
                                                  radius=PANEL["radius"], fill=255)
    above = Image.new("L", (W, H), 0)
    ImageDraw.Draw(above).rectangle((0, 0, W, PANEL["top"] + 2), fill=255)
    pop = Image.fromarray((np.asarray(mlayer, np.float32) * np.asarray(above, np.float32) / 255).astype(np.uint8))
    out = canvas_im.copy()
    out.paste(layer, (0, 0), panel_mask)
    # thin light border so the panel edge reads on dark footage, then the head goes over that edge
    ImageDraw.Draw(out).rounded_rectangle((PANEL["x0"], PANEL["top"], PANEL["x1"], H + PANEL["radius"]),
                                          radius=PANEL["radius"], outline=BORDER, width=3)
    out.paste(layer, (0, 0), pop)
    return out


def main(out_path, events_path):
    plan = json.load(open(PLAN))
    for p in plan:
        p["frame"] = int(np.ceil(p["start"] * FPS - 1e-6))
    alphas = [qp.word_alpha(p["text"], p["class"]) for p in plan]
    views = {v: read_frames(path) for v, path in VIEWS.items()}
    bg = canvas_background()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-vf", "scale=out_color_matrix=bt709", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    for i in range(N):
        v = view_at(i)
        if v == "canvas":
            m = np.asarray(Image.open(f"{MASKS}/m{i + MASK_OFFSET:05d}.png"))
            pic = panel_composite(draw_canvas(i, bg), views["wide"][i], m)
            base = np.asarray(pic).astype(np.float32) / 255
            zoom = 1.0
        else:
            base = views[v][i].astype(np.float32) / 255
            zoom = 1.0
        # captions (Prime: white/pink, Difference blend); centred, or under the card in canvas mode
        cur = [k for k, p in enumerate(plan) if p["frame"] <= i]
        if cur:
            k = cur[-1]
            p, a = plan[k], alphas[k]
            fill = PINK if p["fill"] == "PINK" else WHITE
            h, w = a.shape
            yc = CAPTION_Y_CANVAS if v == "canvas" else H // 2
            x0, y0 = (W - w) // 2, int(yc - h / 2)
            ys, xs = slice(max(0, y0), min(H, y0 + h)), slice(max(0, x0), min(W, x0 + w))
            aa = a[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0][..., None]
            bgr = base[ys, xs]
            base[ys, xs] = bgr * (1 - aa) + np.abs(np.array(fill, np.float32) - bgr) * aa
            if p["class"] == "script" and v != "canvas":
                kf = i - p["frame"]
                zoom = 1.0 + qp.ZOOM_AMOUNT * (1.0 - np.exp(-kf / qp.ZOOM_TAU))
        if i in NEGATIVE:
            base = 1.0 - base
        frame = Image.fromarray((np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8))
        if zoom > 1.0005:
            cw, ch = W / zoom, H / zoom
            frame = frame.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()

    # sound events for sfx_v12.py (times = first frame of the event)
    ev = []
    for p in plan:
        ev.append({"t": p["frame"] / FPS, "type": "shutter"})                        # every caption
        if p["class"] == "script":
            ev.append({"t": p["frame"] / FPS, "type": "hit", "size": 1.25 if p is plan[-1] else 0.9})
    for f0 in (15, 99, 106, 114):
        ev.append({"t": f0 / FPS, "type": "flash"})                                  # quick cuts
    for f0 in (CANVAS_IN, CANVAS_OUT):
        ev.append({"t": f0 / FPS, "type": "whoosh"})                                 # scene to scene, peak on the cut
    ev.append({"t": TITLE_AT / FPS, "type": "glitch"})                                # text graphic pops up
    ev += [{"t": f / FPS, "type": "click"} for f in POPS]                             # elements pop up
    ev.append({"t": 119 / FPS, "type": "riser", "length": 1.0})                       # build into "Willen"
    json.dump(sorted(ev, key=lambda e: e["t"]), open(events_path, "w"), indent=1)
    print(f"rendered {N} frames -> {out_path}; {len(ev)} sound events -> {events_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
