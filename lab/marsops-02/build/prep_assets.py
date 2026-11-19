"""Prepare the teaser's archival images for the render page (build/img/): crops at working size, the solar-flare film's
frames, the Pacioli capitals, and the outline of the spacesuit traced from NASA's drawing (so the neon shield drawn
around it follows the real suit). Run with the QC venv python (needs numpy, Pillow, OpenCV):
    python -I prep_assets.py
"""
import json
import os
import subprocess

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
M = "/home/user/SocialMediaMarketingSkill/lab/marsops-02"
B = f"{M}/broll"
OUT = f"{M}/build/img"
os.makedirs(OUT, exist_ok=True)
meta = {}


def save(im, name, q=92):
    im.save(f"{OUT}/{name}", quality=q)
    meta[name] = im.size


# 1. Leonardo, aerial screw (manuscript B, f. 83v; Ravaisson-Mollien facsimile 1883): the manuscript page without the
#    printed facsimile margins
im = Image.open(f"{B}/leonardo_msB_f83v_aerial_screw/ravaisson1883_vol2_n340.jpg").convert("RGB")
w, h = im.size
leo = im.crop((int(0.115 * w), int(0.075 * h), int(0.885 * w), int(0.80 * h)))
leo = leo.resize((1800, int(leo.height * 1800 / leo.width)), Image.LANCZOS)
save(leo, "leo.jpg")

# 2. Mars (Viking mosaic, PIA00407)
mars = Image.open(f"{B}/mars_viking_pia00407/PIA00407.tif").convert("RGB").resize((2000, 2000), Image.LANCZOS)
save(mars, "mars.jpg")

# 3. The Sun's X1.9 flare (SDO/AIA 171 A, 19 June 2025): 8.7-11.1 s of the film at 30 fps, the disk only
os.makedirs(f"{OUT}/sdo", exist_ok=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "8.7", "-t", "2.4", "-i",
                f"{B}/sdo_x19_flare_2025_171/Flares_20250619_AR14114X19_AIA.0171A.PSF_2048p30.mp4",
                "-vf", "crop=1800:1800:124:124,scale=1200:1200", "-q:v", "3", f"{OUT}/sdo/f%03d.jpg"], check=True)
meta["sdo_frames"] = len(os.listdir(f"{OUT}/sdo"))

# 4. NASA's Shuttle EMU drawing (S91-37110): the whole front view of the suit, helmet to boots, with its labels
im = Image.open(f"{B}/emu_cutaway_drawing_S91-37110/S91-37110_orig.jpg").convert("L")
w, h = im.size
ea = np.asarray(im).copy()
# retouch the NASA logotype on the chest (no implied endorsement); box in the original's pixels
X0, Y0, X1, Y1 = 3684, 2526, 3905, 2627
meta["nasa_logo_retouched_box_orig"] = [X0, Y0, X1, Y1]
ea[Y0:Y1, X0:X1] = 255
im = Image.fromarray(ea)
EB = (0.50, 0.13, 0.95, 0.715)                                   # crop, as fractions of the original
emu = im.crop((int(EB[0] * w), int(EB[1] * h), int(EB[2] * w), int(EB[3] * h)))
emu = emu.resize((1400, int(emu.height * 1400 / emu.width)), Image.LANCZOS)
# the back view's labels cut by the crop's left edge (word ends with no beginning): whitened. Boxes measured on a
# 500 px wide view of this crop
ea = np.asarray(emu).copy(); sx, sy = emu.width / 500, emu.height / 845
for x0, y0, x1, y1 in [(0, 180, 95, 222), (0, 252, 28, 274), (0, 314, 62, 348), (0, 364, 98, 398), (0, 411, 44, 448), (0, 494, 14, 512)]:
    ea[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)] = 255
emu = Image.fromarray(ea)
meta["emu_cut_labels_whitened"] = True
save(emu.convert("RGB"), "emu.jpg")
# the shield: a smooth line around the figure (helmet to boots), points placed on the original (fractions), drawn by
# two pens from the top in the render
SH = [(0.69, 0.150), (0.76, 0.158), (0.815, 0.19), (0.838, 0.26), (0.833, 0.36), (0.818, 0.45), (0.803, 0.55),
      (0.792, 0.64), (0.762, 0.703), (0.69, 0.722), (0.618, 0.703), (0.582, 0.64), (0.570, 0.55), (0.556, 0.45),
      (0.560, 0.36), (0.582, 0.27), (0.615, 0.19), (0.645, 0.16)]
meta["emu_shield"] = [[round((x - EB[0]) / (EB[2] - EB[0]), 4), round((y - EB[1]) / (EB[3] - EB[1]), 4)] for x, y in SH]
chk = emu.convert("RGB").resize((500, int(500 * emu.height / emu.width)))
from PIL import ImageDraw
dr = ImageDraw.Draw(chk)
pts = [(x * 500, y * chk.height) for x, y in meta["emu_shield"]]
dr.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
chk.save(f"{M}/build/emu_shield_check.png")

# 5. NASA Mars habitat concept (PIA23302)
hab = Image.open(f"{B}/mars_habitat_concept_PIA23302/PIA23302_orig.jpg").convert("RGB")
hab = hab.resize((3000, int(hab.height * 3000 / hab.width)), Image.LANCZOS)
save(hab, "hab.jpg")

# 6. Pacioli's capitals (Divina proportione, 1509; The Met): the letter and its construction, square crop
for L, f in [("M", "letter_M_DP156952"), ("A", "letter_A_DP156941"), ("R", "letter_R_DP156958"), ("S", "letter_S_DP156959"),
             ("O", "letter_O_DP156955"), ("P", "letter_P_DP156956")]:
    im = Image.open(f"{B}/met_pacioli_divina_proportione_letters/{f}.jpg").convert("L")
    g = np.asarray(im)
    ys, xs = np.where(g < 120)
    # the printed letter block sits in the upper part, the text below: take the largest ink cluster's square
    top_part = ys < np.percentile(ys, 70)
    x0, x1 = np.percentile(xs[top_part], [1, 99]); y0, y1 = np.percentile(ys[top_part], [1, 99])
    side = max(x1 - x0, y1 - y0) * 1.06; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    paper = int(np.median(g))
    canvas = Image.new("L", (int(side), int(side)), paper)          # pad with paper colour (no dark scan edges)
    x0c, y0c = int(cx - side / 2), int(cy - side / 2)
    part = im.crop((max(0, x0c), max(0, y0c), min(im.width, x0c + int(side)), min(im.height, y0c + int(side))))
    canvas.paste(part, (max(0, -x0c), max(0, -y0c)))
    sq = canvas.resize((700, 700), Image.LANCZOS)
    save(sq.convert("RGB"), f"pacioli_{L}.jpg")

json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
open(f"{OUT}/meta.js", "w").write("window.META = " + json.dumps(meta) + ";\n")   # for the render page (file:// cannot fetch)
print({k: v for k, v in meta.items() if k != "emu_shield"})
