"""Prepare the v3 teaser's images (build/img/): the starfield (ESO Milky Way, rotated for a vertical frame), the Mars
base band (NASA PIA23302), the Sun's eruption frames (NASA SDO 304 A, 7 June 2011), and the measurements the render
page needs (meta.js). Run with the QC venv python: python -I prep_v3.py
"""
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None
M = "/home/user/SocialMediaMarketingSkill/lab/marsops-03"
OUT = f"{M}/build/img"
meta = {}

# 1. The Milky Way (ESO/S. Brunier, GigaGalaxy Zoom; the site's own copy): the band crossing the frame on a diagonal,
#    with room to drift; darkened toward the site's navy
mw = Image.open(f"{M}/site_assets/milky-way-6k.jpg").convert("RGB")
w, h = mw.size
band = mw.rotate(-58, resample=Image.BICUBIC, expand=True)                          # the whole panorama (no empty corners)
bw, bh = band.size
# keep the middle 1400 x 2480 (covers 1080x1920 with room for a slow drift)
cx, cy = bw // 2, bh // 2
band = band.crop((cx - 700, cy - 1240, cx + 700, cy + 1240))
a = np.asarray(band).astype(np.float32) / 255
navy = np.array([3, 8, 19], np.float32) / 255
a = navy + (a - navy).clip(0) * 0.9
Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8)).save(f"{OUT}/stars.jpg", quality=90)
meta["stars"] = [1400, 2480]

# 2. The Mars base (NASA PIA23302): a band 1500 px tall across the frame, the habitats centred
hab = Image.open(f"{OUT}/hab_orig.jpg").convert("RGB")
s = 1500 / hab.height
hw = int(hab.width * s)
hab = hab.resize((hw, 1500), Image.LANCZOS)
cxh = int(0.555 * hw)
hab = hab.crop((cxh - 540, 0, cxh + 540, 1500))
hab.save(f"{OUT}/hab.jpg", quality=92)
# where the base sits in this crop (fractions of the original: habitats x 0.37-0.73, ground line y ~0.60)
meta["hab"] = {"x0": (0.37 * hw - (cxh - 540)), "x1": (0.73 * hw - (cxh - 540)), "ground": 0.605 * 1500, "top": 0.49 * 1500}

# 3. The Sun's eruption (NASA SDO/AIA 304 A, SVS 3838): film 1.4-4.9 s, the full disk
os.makedirs(f"{OUT}/sun", exist_ok=True)
src = "/home/user/SocialMediaMarketingSkill/lab/marsops-02/broll/sdo_eruption_2011_304/MonsterFilament_304A_4096p30_h265.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.4", "-t", "3.5", "-i", src, "-vf", "crop=3900:3900:98:98,scale=760:760",
                "-q:v", "3", f"{OUT}/sun/f%03d.jpg"], check=True)
n = len(os.listdir(f"{OUT}/sun"))
g = np.asarray(Image.open(f"{OUT}/sun/f001.jpg").convert("L")).astype(np.float32)
disk = g > 0.5 * np.median(g[g > 20])                                               # the disk, not the faint corona
rows, cols = np.where(disk.any(1))[0], np.where(disk.any(0))[0]
meta["sun"] = {"frames": n, "size": 760, "cx": float(cols[0] + cols[-1]) / 2, "cy": float(rows[0] + rows[-1]) / 2,
               "r": float(rows[-1] - rows[0]) / 2}

open(f"{OUT}/meta.js", "w").write("window.META = " + json.dumps(meta) + ";\n")
print(json.dumps(meta, indent=1))
