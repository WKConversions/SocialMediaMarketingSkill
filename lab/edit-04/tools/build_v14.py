"""Brooklyn Bridge -> WKC story edit (clips 1111-4444), version 1.

Picture per output frame comes from one of: a talking-head view (wide/medium/close of the clip that is
speaking, graded), an archival insert (real footage, black & white film treatment) or a canvas moment.
Prime captions on top (centred, Difference blend; under the card in canvas mode). Sound events are
written for sfx_v14.py. Pacing (uniform pauses) comes from out/edl.json (tools/pacing.py).
Usage: python3 build_v14.py
"""
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/edit-03/tools")
import quick_prime as qp  # noqa: E402
from build_v13 import background, font, glitch_text, BORDER, CARD  # noqa: E402

E = "/home/user/SocialMediaMarketingSkill/lab/edit-04"
W, H, FPS = 1080, 1920, 30
EDL = json.load(open(f"{E}/out/edl.json"))
CFG = json.load(open(f"{E}/out/edit_v14.json"))          # creative decisions (views, inserts, canvases, captions)
SPEECH_END = EDL["duration"]
END_CARD = CFG["end_card_s"]
N = int(round((SPEECH_END + END_CARD) * FPS))
rng = np.random.default_rng(3)
PUNCH = 0.04          # client: don't push far into the face (was 0.12)


def seg_at(t):
    for s in EDL["segments"]:
        if s["out_start"] - 1e-6 <= t < s["out_end"] - 1e-6:
            return s
    return EDL["segments"][-1]


class Reader:
    """Streams frames forward from a video (decoding on demand keeps memory flat); restarts on a backward seek."""

    def __init__(self, path, start=0.0):
        self.path, self.start, self.proc, self.idx, self.last = path, start, None, -1, None

    def _open(self):
        if self.proc:
            self.proc.kill()
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(self.start), "-i", self.path, "-vf",
                                      "scale=in_color_matrix=bt709:out_color_matrix=bt709", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                     stdout=subprocess.PIPE)
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


def read_video(path, n=None, start=0.0):
    args = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", path, "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709",
            "-f", "rawvideo", "-pix_fmt", "rgb24"]
    if n:
        args += ["-frames:v", str(n)]
    raw = subprocess.run(args + ["-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


# ---------------------------------------------------------------- grades
def grade_talking(x, era):
    """era 'history': cooler, a little desaturated and contrasty (serious); 'now': warm, clean, trustworthy."""
    lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sat = 0.78 if era == "history" else 1.02
    x = lum[..., None] + (x - lum[..., None]) * sat
    x = np.clip((x - 0.5) * 1.12 + 0.5, 0, 1)                      # gentle contrast
    x = x * x * (3 - 2 * x) * 0.35 + x * 0.65                       # soft S-curve
    sh = (1 - lum)[..., None] ** 2                                   # shadows weight
    hi = lum[..., None] ** 1.5
    if era == "history":
        x = x + sh * np.array([-0.020, 0.0, 0.025], np.float32) + hi * np.array([0.01, 0.005, -0.01], np.float32)
    else:
        x = x + sh * np.array([-0.012, 0.0, 0.016], np.float32) + hi * np.array([0.03, 0.012, -0.02], np.float32)
    yy, xx = np.ogrid[:H, :W]
    vig = 1 - 0.22 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 1.6)) ** 2)
    return np.clip(x * vig[..., None], 0, 1)


def grade_archival(x, i):
    g = x @ np.array([0.299, 0.587, 0.114], np.float32)
    g = np.clip((g - 0.5) * 1.25 + 0.5 + rng.normal(0, 0.012), 0, 1)          # contrast + per-frame flicker
    g = g + rng.normal(0, 0.045, g.shape).astype(np.float32)                    # film grain
    yy, xx = np.ogrid[:H, :W]
    vig = 1 - 0.45 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    g = np.clip(g * vig, 0, 1)
    return np.stack([g * 1.02, g * 0.98, g * 0.92], axis=-1).clip(0, 1)         # faint warm print tone


# ---------------------------------------------------------------- canvases
def canvas(kind, i, f0, bg, extra):
    im = bg.copy()
    if kind == "equation":
        im = glitch_text(im, ["Marketing"], i, f0 + 2, 470, px=150)
        im = glitch_text(im, ["= Sales"], i, f0 + 2, 690, px=150)
    elif kind == "showcase":
        im = glitch_text(im, ["We showcase", "your product"], i, f0 + 2, 160, px=84)
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = 190, 470, 890, 1150
        d.rounded_rectangle((x0, y0, x1, y1), radius=34, fill=CARD, outline=BORDER, width=3)
        if extra is not None:                                        # website capture inside the card
            shot = Image.fromarray(extra.get(max(0, i - f0))).resize((x1 - x0 - 24, int((x1 - x0 - 24) * H / W)), Image.LANCZOS)
            shot = shot.crop((0, 0, shot.width, y1 - y0 - 24))
            m = Image.new("L", shot.size, 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, shot.width, shot.height), radius=24, fill=255)
            im.paste(shot, (x0 + 12, y0 + 12), m)
        else:
            d.text(((x0 + x1) / 2, (y0 + y1) / 2), "wkconversions.com", font=font(58, 750), fill=(235, 235, 235), anchor="mm")
    return im


def panel_plain(canvas_im, frame_rgb):
    """Speaker in the bottom panel (v1: no head pop-out yet)."""
    s, top, x0, x1, r = 0.75, 1390, 135, 945, 46
    fw, fh = int(W * s), int(H * s)
    small = Image.fromarray(frame_rgb).resize((fw, fh), Image.LANCZOS)
    layer = Image.new("RGB", (W, H))
    layer.paste(small, ((W - fw) // 2, int(top - 0.36 * fh)))
    pmask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(pmask).rounded_rectangle((x0, top, x1, H + r), radius=r, fill=255)
    out = canvas_im.copy()
    out.paste(layer, (0, 0), pmask)
    ImageDraw.Draw(out).rounded_rectangle((x0, top, x1, H + r), radius=r, outline=BORDER, width=3)
    return out


def end_card(i, f0, bg, logo):
    im = bg.copy()
    if logo is not None:
        lg = logo.copy()
        lg.thumbnail((560, 300))
        im.paste(lg, ((W - lg.width) // 2, 700 - lg.height // 2), lg if lg.mode == "RGBA" else None)
    im = glitch_text(im, ["wkconversions.com"], i, f0, 860, px=78)
    return im


# ---------------------------------------------------------------- main
def window(lst, t):
    for it in lst:
        if it["start"] - 1e-6 <= t < it["end"] - 1e-6:
            return it
    return None


def main():
    caps = CFG["captions"]
    for c in caps:
        c["frame"] = max(0, int(np.ceil(c["start"] * FPS - 1e-6)) - 1)      # one frame of lead
    alphas = [qp.word_alpha(c["text"], c["cls"]) for c in caps]
    views = {(clip, v): Reader(f"{E}/views/{clip}_{v}.mp4") for clip in ("1111", "2222", "3333", "4444")
             for v in ("wide", "medium", "close")}
    inserts = {k: Reader(ins["file"], ins.get("src", 0.0)) for k, ins in enumerate(CFG["inserts"])}
    web = Reader(CFG["website"]) if CFG.get("website") else None
    logo = Image.open(CFG["logo"]).convert("RGBA") if CFG.get("logo") else None
    bg = background()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-vf", "scale=out_color_matrix=bt709", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-movflags", "+faststart", f"{E}/out/v14_noaudio.mp4"], stdin=subprocess.PIPE)
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
            v = window(CFG["views"], t)["view"]
            frame = views[(seg["clip"], v)].get(sf)
            era = "history" if seg["clip"] in ("1111", "2222") else "now"
            ins = window(CFG["inserts"], t)
            cv = window(CFG["canvases"], t)
            if ins is not None:
                k = CFG["inserts"].index(ins)
                j = int(round((t - ins["start"]) * FPS))
                src = inserts[k].get(j).astype(np.float32) / 255
                z = 1 + 0.06 * (t - ins["start"]) / max(0.1, ins["end"] - ins["start"])   # slow push-in
                if z > 1.001:
                    img = Image.fromarray((src * 255).astype(np.uint8))
                    cw, ch = W / z, H / z
                    src = np.asarray(img.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))).astype(np.float32) / 255
                pic = grade_archival(src, i)
            elif cv is not None:
                f0 = int(round(cv["start"] * FPS))
                head = grade_talking(frame.astype(np.float32) / 255, era)
                c_im = canvas(cv["kind"], i, f0, bg, web if cv["kind"] == "showcase" else None)
                pic = np.asarray(panel_plain(c_im, (head * 255).astype(np.uint8))).astype(np.float32) / 255
                canvas_on = True
            else:
                pic = grade_talking(frame.astype(np.float32) / 255, era)
        base = pic
        zoom = 1.0
        cur = [k for k, c in enumerate(caps) if c["frame"] <= i]
        if cur and t < SPEECH_END:
            c, a = caps[cur[-1]], alphas[cur[-1]]
            h, w = a.shape
            yc = 1245 if canvas_on else H // 2
            x0, y0 = (W - w) // 2, int(yc - h / 2)
            ys, xs = slice(max(0, y0), min(H, y0 + h)), slice(max(0, x0), min(W, x0 + w))
            aa = a[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0][..., None]
            fill = np.array(qp.PINK if c.get("fill") == "PINK" else qp.WHITE, np.float32)
            region = base[ys, xs]
            base[ys, xs] = region * (1 - aa) + np.abs(fill - region) * aa
            if c["cls"] == "script" and not canvas_on:
                zoom = 1.0 + PUNCH * (1.0 - np.exp(-(i - c["frame"]) / qp.ZOOM_TAU))
        if i in neg:
            base = 1.0 - base
        out = Image.fromarray((np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8))
        if zoom > 1.0005:
            cw, ch = W / zoom, H / zoom
            out = out.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
        enc.stdin.write(out.tobytes())
    enc.stdin.close()
    enc.wait()
    json.dump([{"text": c["text"], "frame": c["frame"], "class": c["cls"]} for c in caps], open(f"{E}/out/v14_captions.json", "w"), indent=1)
    print(f"rendered {N} frames ({N / FPS:.2f} s)")


if __name__ == "__main__":
    main()
