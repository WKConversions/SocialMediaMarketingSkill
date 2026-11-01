"""Checks on the finished Wright 1903 classy edit (run with the venv python: needs torch/transformers for alignment).

1. Caption timing: re-align the transcript on the final voice (the clean voice on the video timeline); every caption
   must become visible 0-2 frames before its first word.
2. Shutter: its biggest wave must sit on each caption's (and label's) first frame (cross-correlation of the final mix
   against the shutter-only track, both high-passed above 2 kHz).
Usage: python verify_v19.py FINAL.mp4 CAPTIONS.json SHUTTER_ONLY.wav VOICE_ON_SAME_TIMELINE.wav
"""
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/lab/caption-style-01/tools")
import align_words  # noqa: E402

FPS, SR = 30, 48000
E = "/home/user/SocialMediaMarketingSkill/lab/edit-09"
TRANSCRIPT = " ".join(w["word"] for w in json.load(open(f"{E}/out/edl.json"))["words"])
FIRST_WORD = {"In 1903": "in"}


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(float)


def hp(x, fc=2000):
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / SR) < fc] = 0
    return np.fft.irfft(X, len(x))


def main(final, captions_path, shutter_only, voice):
    caps = json.load(open(captions_path))
    out_words = "/tmp/claude-0/v19_final_words.json"
    align_words.main(voice, TRANSCRIPT, out_words, "jonatasgrosman/wav2vec2-large-xlsr-53-english")
    words = json.load(open(out_words))
    wi, bad = 0, 0
    print("\ncaption                        frame   word starts   lead")
    for c in caps:
        if c.get("label"):
            continue
        first = FIRST_WORD.get(c["text"], c["text"].split()[0].lower().strip(",?"))
        while wi < len(words) and words[wi]["word"] != first:
            wi += 1
        ws = words[wi]["start"]
        lead = ws - c["frame"] / FPS
        ok = -0.5 / FPS <= lead <= 2.5 / FPS
        bad += not ok
        print(f"{c['text']:30s} f{c['frame']:4d}   {ws:6.3f}s   {lead * 1000:+5.0f} ms {'' if ok else '  <-- OUT OF RANGE'}")
        wi += 1
    fin, ref = hp(load(final)), hp(load(shutter_only))
    offs = []
    for c in caps:
        ft = c["frame"] / FPS
        a, b = int(max(0, ft - 0.03) * SR), int((ft + 0.03) * SR)
        r, best, lb = ref[a:b], -1, 0
        for lag in range(-480, 481, 2):
            if a + lag < 0:
                continue
            f = fin[a + lag:b + lag]
            cc = np.dot(f, r) / (np.linalg.norm(f) * np.linalg.norm(r) + 1e-12)
            if cc > best:
                best, lb = cc, lag
        offs.append(lb / SR * 1000)
    print("\nshutter biggest wave vs caption frame (ms):", " ".join(f"{o:+.1f}" for o in offs))
    print(f"\ncaptions out of range: {bad}; shutters off by more than 1 ms: {sum(abs(o) > 1 for o in offs)}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
