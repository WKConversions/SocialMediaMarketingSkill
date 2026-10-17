"""Diagnostic plots for the caption shutter sound.

1. The client's camera_shutter.wav on its own: waveform, loudness envelope and spectrogram, with the
   individual clicks, the part that is audible, the slice the renderer uses and the sync point marked.
2. The rendered video: voice and effects on one timeline, with the frames where captions appear.

Usage: python plot_shutter.py SHUTTER.wav SFX_TRACK.wav FINAL.mp4 PLAN.json OUT_DIR
"""
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(path, sr):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(float)


def env_db(x, sr, win_ms=1.0):
    w = max(1, int(sr * win_ms / 1000))
    e = np.array([np.abs(x[i:i + w]).max() for i in range(0, len(x) - w, w)])
    return 20 * np.log10(e / (np.abs(x).max() + 1e-12) + 1e-9)


def clicks(edb, level=-24, min_gap=12):
    """One click per burst: the loudest millisecond of each stretch that rises above `level` dB."""
    out, i = [], 0
    while i < len(edb):
        if edb[i] > level:
            j = i
            while j < len(edb) and edb[j] > level - 6:
                j += 1
            k = i + int(np.argmax(edb[i:j]))
            if not out or k - out[-1] >= min_gap:
                out.append(k)
            elif edb[k] > edb[out[-1]]:
                out[-1] = k
            i = j
        else:
            i += 1
    return out


def plot_sample(path, out_png):
    sr = 48000
    x = load(path, sr)
    t_ms = np.arange(len(x)) / sr * 1000
    edb = env_db(x, sr)
    ck = clicks(edb)
    peak_ms = int(np.argmax(np.abs(x)) / sr * 1000)
    audible = np.where(edb > -40)[0]
    a0, a1 = audible[0], audible[-1]
    trim0, trim1, slice_click = 95, 300, 185
    used_onset = trim0 + int(np.argmax(edb[trim0:] > -30))   # first sample within 30 dB of the peak, after the trim

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1.1]})
    fig.suptitle("Your camera_shutter.wav: where the sound actually is", x=0.06, ha="left", fontsize=15, color=INK, weight="bold")
    fig.text(0.06, 0.935, f"File is {len(x)/sr*1000:.0f} ms long. Audible (within 40 dB of its peak) only from {a0} to {a1} ms. "
             f"The rest is near-silence.", fontsize=11, color=INK2)

    a = ax[0]
    a.axvspan(a0, a1, color=BLUE, alpha=0.07, lw=0)
    a.plot(t_ms, x, color=BLUE, lw=0.7)
    a.set_ylabel("waveform")
    a.text(a0 + 2, a.get_ylim()[1] * 0.85, "audible part", color=INK2, fontsize=10)

    b = ax[1]
    b.plot(np.arange(len(edb)), edb, color=BLUE, lw=1.6)
    b.axhline(-30, color=MUTED, lw=1, ls=(0, (4, 3)))
    b.text(2, -28, "-30 dB: the level I treat as 'the sound has started'", color=INK2, fontsize=9)
    b.set_ylim(-75, 3)
    b.set_ylabel("loudness (dB below peak)")
    for i, c in enumerate(ck, 1):
        for axx in ax[:2]:
            axx.axvline(c, color=INK2, lw=0.8, alpha=0.6)
        lab = f"click {i}\n{c} ms" + ("\n(loudest =\nthe 'shutter')" if abs(c - peak_ms) <= 3 else "")
        b.annotate(lab, (c, edb[c]), xytext=(c + 4, edb[c] - 30 - 8 * (i % 2)), fontsize=9, color=INK,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    c = ax[2]
    c.specgram(x, NFFT=512, Fs=sr / 1000, noverlap=448, cmap="Blues", vmin=-120, scale="dB")
    c.set_ylabel("frequency (kHz)")
    c.set_ylim(0, 16)
    c.set_xlabel("time inside the file (ms)")
    c.grid(False)

    # what the renderer does with it
    for axx in ax:
        axx.axvline(trim0, color=ORANGE, lw=1.6)
        axx.axvline(trim1, color=ORANGE, lw=1.6)
    ax[0].annotate("renderer cuts the file here", (trim0, ax[0].get_ylim()[0] * 0.9), xytext=(trim0 - 88, ax[0].get_ylim()[0] * 0.9),
                   fontsize=9, color=INK, va="center", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax[0].annotate("...and here", (trim1, ax[0].get_ylim()[0] * 0.9), xytext=(trim1 + 8, ax[0].get_ylim()[0] * 0.9),
                   fontsize=9, color=INK, va="center", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax[1].plot([used_onset], [edb[used_onset]], "o", ms=9, color=ORANGE, mec=SURFACE, mew=2, zorder=5)
    ax[1].annotate(f"SYNC POINT ({used_onset} ms):\nthis moment lands on the\nframe the caption appears",
                   (used_onset, edb[used_onset]), xytext=(used_onset - 95, -12), fontsize=10, color=INK, weight="bold",
                   arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    ax[1].axvline(slice_click, color=ORANGE, lw=1, ls=(0, (2, 2)))
    ax[1].text(slice_click + 2, -72, f"'click only' variant starts here ({slice_click} ms)", fontsize=9, color=INK2)
    ax[2].set_xlim(0, len(x) / sr * 1000)
    fig.tight_layout(rect=(0.03, 0, 1, 0.93))
    fig.savefig(out_png, dpi=110)
    return {"length_ms": round(len(x) / sr * 1000), "audible_ms": [int(a0), int(a1)], "clicks_ms": ck,
            "loudest_ms": peak_ms, "sync_point_ms": used_onset}


def plot_video(sfx_path, final_path, plan_path, out_png):
    sr = 16000
    mix = load(final_path, sr)
    sfx = load(sfx_path, sr)
    n = min(len(mix), len(sfx))
    t = np.arange(n) / sr
    plan = json.load(open(plan_path))
    fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle("In the video: when captions appear and when the shutter sounds play", x=0.05, ha="left",
                 fontsize=15, color=INK, weight="bold")
    ax[0].plot(t, mix[:n], color=MUTED, lw=0.5)
    ax[0].set_ylabel("final audio\n(voice + effects)")
    ax[1].plot(t, sfx[:n], color=BLUE, lw=0.6)
    ax[1].set_ylabel("effects only")
    ax[1].set_xlabel("time in the video (s)")
    for p in plan:
        f = int(np.ceil(p["start"] * 30 - 1e-6))
        ft = f / 30
        sounded = p.get("sfx") is not None
        for axx in ax:
            axx.axvline(ft, color=ORANGE if sounded else GRID, lw=1.6 if sounded else 1.2, zorder=0)
        ax[0].text(ft + 0.01, ax[0].get_ylim()[1] * 0.92, p["text"] + ("" if sounded else "\n(silent)"), fontsize=9,
                   color=INK if sounded else MUTED, rotation=0, va="top")
    ax[1].text(0.0, ax[1].get_ylim()[0] * 0.95, "orange line = frame a caption appears with a sound;  grey line = caption without sound",
               fontsize=9, color=INK2, va="bottom")
    ax[1].set_xlim(0, n / sr)
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    fig.savefig(out_png, dpi=110)


def plot_alignment(shutter_only_path, plan_path, out_png, fps=30):
    sr = 48000
    x = load(shutter_only_path, sr)
    plan = [p for p in json.load(open(plan_path)) if p.get("sfx")]
    fig, axes = plt.subplots(1, len(plan), figsize=(3.1 * len(plan), 3.6), sharey=True)
    fig.suptitle("Biggest wave of the shutter vs. the first frame of each caption", x=0.02, ha="left",
                 fontsize=14, color=INK, weight="bold")
    rows = []
    for a, p in zip(np.atleast_1d(axes), plan):
        ft = np.ceil(p["start"] * fps - 1e-6) / fps
        i0, i1 = int((ft - 0.12) * sr), int((ft + 0.12) * sr)
        seg = x[max(0, i0):i1]
        tt = (np.arange(max(0, i0), max(0, i0) + len(seg)) / sr - ft) * 1000
        a.plot(tt, seg, color=BLUE, lw=0.7)
        pk = int(np.argmax(np.abs(seg)))
        off = tt[pk]
        rows.append((p["text"], off))
        a.axvline(0, color=ORANGE, lw=1.8)
        a.plot([off], [seg[pk]], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2, zorder=5)
        a.set_title(f'{p["text"]}\nbiggest wave {off:+.1f} ms', fontsize=10, color=INK)
        a.set_xlim(-120, 120)
        a.set_xlabel("ms from caption frame")
    np.atleast_1d(axes)[0].set_ylabel("shutter sound")
    fig.text(0.02, 0.01, "orange line = first frame the caption is visible;  dot = biggest wave of the shutter", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(out_png, dpi=110)
    return rows


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--align":
    for name, off in plot_alignment(sys.argv[2], sys.argv[3], sys.argv[4]):
        print(f"{name:11s} biggest wave {off:+.1f} ms from caption frame")
    sys.exit(0)

if __name__ == "__main__":
    shutter, sfx, final, plan, out_dir = sys.argv[1:6]
    info = plot_sample(shutter, f"{out_dir}/shutter_analysis.png")
    plot_video(sfx, final, plan, f"{out_dir}/shutter_in_video.png")
    print(json.dumps(info))
