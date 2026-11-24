"""Cut the silences out of the take, extract the graded frames of both angles on the edit's timeline, and cut the
phone audio to match.  python3 -I prep.py   -> build/v/p_%04d.jpg (phone), build/v/a_%04d.jpg (angle 2), build/voice.wav, build/edl.json
The second angle is shifted by the measured audio offset (src/sync.json), so both angles match the phone's sound."""
import json, os, subprocess
E = "/home/user/SocialMediaMarketingSkill/lab/edit-13"; FPS = 30
OFF = json.load(open(f"{E}/src/sync.json"))["offset_s"]          # angle2 time = phone time + OFF
SEGS = [(0.83, 142), (11.69, 70), (22.27, 139), (38.87, 53), (48.39, 182)]   # (phone start s, frames): speech + small pads
GRADE_P = "colortemperature=temperature=9000:mix=0.8,selectivecolor=yellows=0 0 -0.55 0:neutrals=0 0 -0.06 0,eq=brightness=0.06:contrast=1.12:saturation=0.9:gamma=1.13"   # high-key and neutral, like the reference: the beige wall goes light grey
GRADE_A = "colortemperature=temperature=8600:mix=0.75,selectivecolor=yellows=0 0 -0.4 0,eq=brightness=0.09:contrast=1.10:saturation=0.92:gamma=1.18,unsharp=5:5:0.6"
os.makedirs(f"{E}/build/v", exist_ok=True)
n = 0; edl = []; parts = []
for k, (t0, nf) in enumerate(SEGS):
    d = nf / FPS
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{E}/src/phone.mov", "-ss", str(t0), "-t", str(d), "-vf",
                    f"fps={FPS},crop=760:1351:151:503,{GRADE_P},scale=1188:2112:flags=lanczos", "-start_number", str(n), "-q:v", "3", f"{E}/build/v/p_%04d.jpg"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{E}/src/angle2.mov", "-ss", str(t0 + OFF), "-t", str(d), "-vf",
                    f"fps={FPS},{GRADE_A},scale=1080:-2:flags=lanczos", "-start_number", str(n), "-q:v", "3", f"{E}/build/v/a_%04d.jpg"], check=True)
    w = f"{E}/build/v/seg{k}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-t", str(d), "-i", f"{E}/src/phone.mov", "-vn", "-ac", "1", "-ar", "48000",   # seek on the INPUT: as an output option the cut comes after the filters, and the fade-out silenced the rest
                    "-af", f"asetpts=PTS-STARTPTS,highpass=f=80,afade=t=in:d=0.012,afade=t=out:st={d - 0.03}:d=0.03", w], check=True)   # reset the clock first: the fade-out time is relative to the cut
    parts.append(w); edl.append({"seg": k, "src": t0, "frames": nf, "out_frame": n, "out_s": round(n / FPS, 3)}); n += nf
# the outro, recorded with the phone only ("Want to read more? Check the link."): same crop and grade
OUTRO = (0.67, 88)
# the outro file is the camera original (cooler, flatter than the first export): its own grade, matched by eye to the rest
GRADE_O = "colortemperature=temperature=9000:mix=0.8,selectivecolor=yellows=0 0 -0.55 0:neutrals=0 0 -0.06 0,colorbalance=rm=0.035:gm=-0.008:bm=-0.02:rh=0.02:gh=-0.005,eq=brightness=0.0:contrast=1.16:saturation=1.02:gamma=1.0"
d = OUTRO[1] / FPS
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{E}/src/phone_full.mov", "-ss", str(OUTRO[0]), "-t", str(d), "-vf",
                f"fps={FPS},crop=760:1351:151:503,{GRADE_O},scale=1188:2112:flags=lanczos", "-start_number", str(n), "-q:v", "3", f"{E}/build/v/p_%04d.jpg"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(OUTRO[0]), "-t", str(d), "-i", f"{E}/src/phone_full.mov", "-vn", "-ac", "1", "-ar", "48000",
                "-af", f"asetpts=PTS-STARTPTS,highpass=f=80,afade=t=in:d=0.012,afade=t=out:st={d - 0.03}:d=0.03", f"{E}/build/outro.wav"], check=True)
edl.append({"seg": "outro", "src": OUTRO[0], "frames": OUTRO[1], "image_frame": n, "out_s": 20.3})
open(f"{E}/build/v/list.txt", "w").write("".join(f"file '{p}'\n" for p in parts))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{E}/build/v/list.txt", "-c:a", "pcm_s16le", f"{E}/build/voice.wav"], check=True)
json.dump({"fps": FPS, "frames": n, "segments": edl, "offset_s": OFF}, open(f"{E}/build/edl.json", "w"), indent=1)
print(n, "frames", n / FPS, "s")
