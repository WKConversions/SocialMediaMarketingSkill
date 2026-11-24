"""Cut the silences out of the take, extract the graded FULL frames of both angles on the edit's timeline (the page
frames them: face centred, as wide as the source allows), and cut the phone audio to match.
    python -I prep.py   (QC venv: numpy)  -> build/v/p_%04d.jpg (phone), a_%04d.jpg (angle 2), voice.wav, outro.wav, edl.json
The second angle is shifted by the measured audio offset (src/sync.json), so both angles match the phone's sound.
Lighting: the phone take's grade is the reference; the second angle and the outro clip are matched to it by measuring
the face and the wall in each (a per-channel gain and offset), so every shot has the same light."""
import json, os, subprocess
import numpy as np

E = "/home/user/SocialMediaMarketingSkill/lab/edit-13"; FPS = 30
OFF = json.load(open(f"{E}/src/sync.json"))["offset_s"]          # angle2 time = phone time + OFF
SEGS = [(0.83, 142), (11.69, 70), (22.27, 139), (38.87, 53), (48.39, 182)]   # (phone start s, frames): speech + small pads
OUTRO = (0.67, 88)                                               # "Want to read more? Check the link." (phone only)
# face centres (fractions of the frame), measured on frames of each recording
FACE = {"phone": (0.57, 0.555), "outro": (0.545, 0.53), "angle2": (0.43, 0.38)}
WALL = {"phone": (0.08, 0.20, 0.35, 0.40), "outro": (0.08, 0.20, 0.35, 0.40), "angle2": (0.66, 0.04, 0.96, 0.24)}   # x0 y0 x1 y1
BASE = "colortemperature=temperature=9000:mix=0.8,selectivecolor=yellows=0 0 -0.55 0:neutrals=0 0 -0.06 0,eq=brightness=0.06:contrast=1.12:saturation=0.9:gamma=1.13"
PRE = {"phone": BASE, "outro": "colortemperature=temperature=9000:mix=0.8,selectivecolor=yellows=0 0 -0.55 0:neutrals=0 0 -0.06 0,eq=contrast=1.16:saturation=1.02",
       "angle2": "colortemperature=temperature=8600:mix=0.75,selectivecolor=yellows=0 0 -0.4 0,eq=brightness=0.09:contrast=1.10:saturation=0.92:gamma=1.18"}
SRC = {"phone": f"{E}/src/phone.mov", "outro": f"{E}/src/phone_full.mov", "angle2": f"{E}/src/angle2.mov"}


def sample(name, times):
    """Mean RGB of the face and of the wall, after the source's base grade."""
    f, w = [], []
    for t in times:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", SRC[name], "-frames:v", "1", "-vf", f"{PRE[name]},scale=200:356", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        a = np.frombuffer(raw, np.uint8).reshape(356, 200, 3).astype(float)
        fx, fy = FACE[name]; x0, y0, x1, y1 = WALL[name]
        f.append(a[int((fy - 0.045) * 356):int((fy + 0.045) * 356), int((fx - 0.07) * 200):int((fx + 0.07) * 200)].mean((0, 1)))
        w.append(a[int(y0 * 356):int(y1 * 356), int(x0 * 200):int(x1 * 200)].mean((0, 1)))
    return np.mean(f, 0), np.mean(w, 0)


ref_f, ref_w = sample("phone", [2, 4, 13, 24, 39, 50, 52])
GRADE = {"phone": PRE["phone"]}; match = {"phone": {"face": ref_f.round(1).tolist(), "wall": ref_w.round(1).tolist()}}
for name, times in (("outro", [1.2, 2.0, 3.0]), ("angle2", [2, 4, 13, 24, 39, 50, 52])):
    f, w = sample(name, times); a = np.sqrt((ref_w / w) * (ref_f / f)); b = np.zeros(3)      # one gain per channel: the mean of what the wall and the face each ask for
    # (face and wall are too close in brightness for a stable gain + offset fit)
    lut = ":".join(f"{c}='clip({a[i]:.4f}*val+{b[i]:.2f},0,255)'" for i, c in enumerate("rgb"))
    GRADE[name] = f"{PRE[name]},lutrgb={lut}"
    match[name] = {"face_before": f.round(1).tolist(), "wall_before": w.round(1).tolist(), "gain": a.round(3).tolist(), "offset": b.round(1).tolist()}
print(json.dumps(match, indent=1))

os.makedirs(f"{E}/build/v", exist_ok=True)
n = 0; edl = []; parts = []
for k, (t0, nf) in enumerate(SEGS):
    d = nf / FPS
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-t", str(d + 0.2), "-i", SRC["phone"], "-frames:v", str(nf), "-vf",
                    f"fps={FPS},{GRADE['phone']},scale=1080:1922:flags=lanczos", "-start_number", str(n), "-q:v", "3", f"{E}/build/v/p_%04d.jpg"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0 + OFF), "-t", str(d + 0.2), "-i", SRC["angle2"], "-frames:v", str(nf), "-vf",
                    f"fps={FPS},{GRADE['angle2']},unsharp=5:5:0.6,scale=1080:1924:flags=lanczos", "-start_number", str(n), "-q:v", "3", f"{E}/build/v/a_%04d.jpg"], check=True)
    w = f"{E}/build/v/seg{k}.wav"       # seek on the INPUT (as an output option the cut comes after the filters and the fade-out silences the rest)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-t", str(d), "-i", SRC["phone"], "-vn", "-ac", "1", "-ar", "48000",
                    "-af", f"asetpts=PTS-STARTPTS,highpass=f=80,afade=t=in:d=0.012,afade=t=out:st={d - 0.03}:d=0.03", w], check=True)
    parts.append(w); edl.append({"seg": k, "src": t0, "frames": nf, "out_frame": n, "out_s": round(n / FPS, 3)}); n += nf
d = OUTRO[1] / FPS
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(OUTRO[0]), "-t", str(d + 0.2), "-i", SRC["outro"], "-frames:v", str(OUTRO[1]), "-vf",
                f"fps={FPS},{GRADE['outro']},scale=1080:1922:flags=lanczos", "-start_number", str(n), "-q:v", "3", f"{E}/build/v/p_%04d.jpg"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(OUTRO[0]), "-t", str(d), "-i", SRC["outro"], "-vn", "-ac", "1", "-ar", "48000",
                "-af", f"asetpts=PTS-STARTPTS,highpass=f=80,afade=t=in:d=0.012,afade=t=out:st={d - 0.03}:d=0.03", f"{E}/build/outro.wav"], check=True)
edl.append({"seg": "outro", "src": OUTRO[0], "frames": OUTRO[1], "image_frame": n, "out_s": 20.3})
open(f"{E}/build/v/list.txt", "w").write("".join(f"file '{p}'\n" for p in parts))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{E}/build/v/list.txt", "-c:a", "pcm_s16le", f"{E}/build/voice.wav"], check=True)
json.dump({"fps": FPS, "frames": n, "segments": edl, "offset_s": OFF, "face": FACE, "light_match": match}, open(f"{E}/build/edl.json", "w"), indent=1)
print(n, "frames", n / FPS, "s")
