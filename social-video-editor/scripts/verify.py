"""Verify the FINISHED file against the plan and the client's rules; write report.json and proof images.

Everything is measured on final.mp4 itself:
  container   frame count, size, frame rate, bt709 tags; audio and video durations within a frame
  loudness    integrated loudness at the style's final target (+- tolerance), sample peak below 0 dBFS
  captions    which caption is on screen in every final frame: each frame is compared (at 1/detect_scale
              size, around the captions) with the clean clip frame composited with each nearby caption
              and with none; the closest wins. The caption changes found this way must be exactly the
              planned first frames.
  timing      the final audio is re-aligned to the transcript; every caption must appear 0-2 frames
              before its first word (+- one alignment frame)
  shutter     the final audio is cross-correlated with the shutter-only track around every caption; the
              shutter's biggest wave must sit within shutter_tolerance_ms of the caption's first frame
  no extras   the final audio minus the voice (rebuilt through the identical chain and limiter with silent
              effects) is the effects part; every sound onset in it must sit at a caption frame (the
              shutter's clicks and the hit), and every caption frame must have one: no whooshes, run-ups or
              other sounds in between
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from align import align
import mix
from common import decode_audio, file_frames, loudness, probe
from plan import spoken_words
from render_captions import apply_zoom, layers, visible, zoom_at

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GREEN, PINK = "#2a78d6", "#eb6834", "#3a9d5d", "#d97a7a"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False})


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, **detail):
        self.items.append(dict(name=name, ok=bool(ok), **detail))
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: " + ", ".join(f"{k}={v}" for k, v in detail.items()
                                                              if not isinstance(v, (list, dict))))

    @property
    def ok(self):
        return all(c["ok"] for c in self.items)


# ------------------------------------------------------------------------------------- captions on screen
def block_mean(img, f):
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    x = img[:h, :w].astype(np.float32)
    return x.reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3)).squeeze()


def small_layer(ly, f):
    """A caption layer at 1/f size: its alpha block-averaged on the frame's own f x f grid."""
    h, w = ly.alpha.shape
    X0, Y0 = ly.x0 // f * f, ly.y0 // f * f
    canvas = np.zeros((math.ceil((ly.y0 + h - Y0) / f) * f, math.ceil((ly.x0 + w - X0) / f) * f), np.float32)
    canvas[ly.y0 - Y0:ly.y0 - Y0 + h, ly.x0 - X0:ly.x0 - X0 + w] = ly.alpha
    return dict(x0=X0 // f, y0=Y0 // f, alpha=block_mean(canvas[..., None], f), layer=ly)


def predict(bg, sl, k, style, fps):
    """Small clean frame + one small caption layer (or None) -> what the finished frame should look like."""
    if sl is None:
        return bg
    out = bg.copy()
    a = sl["alpha"][..., None]
    h, w = a.shape[:2]
    reg = out[sl["y0"]:sl["y0"] + h, sl["x0"]:sl["x0"] + w]
    a = a[:reg.shape[0], :reg.shape[1]]
    out[sl["y0"]:sl["y0"] + h, sl["x0"]:sl["x0"] + w] = reg * (1 - a) + np.abs(sl["layer"].colour - reg) * a
    s = zoom_at(sl["layer"], k / fps, style["zoom"])
    if s > 1.0 + style["zoom"]["min_visible"]:
        out = apply_zoom((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), s, style["zoom"]["origin"]) / 255.0
    return out


def detect_captions(final, clip, plan, style, cfg):
    """For every final frame: index of the caption on screen (-1 = none) and the decision margin."""
    f, fps = cfg["detect_scale"], clip.fps
    lays = layers(plan, style, clip.width, clip.height)
    small = [small_layer(ly, f) for ly in lays]
    # region of interest: every caption's box, grown by the largest zoom about the origin
    zmax = 1 + style["zoom"]["amount"]
    ox, oy = (o * d / f for o, d in zip(style["zoom"]["origin"], (clip.width, clip.height)))
    y0 = min(oy + (s["y0"] - oy) * zmax for s in small)
    y1 = max(oy + (s["y0"] + s["alpha"].shape[0] - oy) * zmax for s in small)
    x0 = min(ox + (s["x0"] - ox) * zmax for s in small)
    x1 = max(ox + (s["x0"] + s["alpha"].shape[1] - ox) * zmax for s in small)
    roi = (slice(max(0, int(y0)), int(math.ceil(y1))), slice(max(0, int(x0)), int(math.ceil(x1))))
    K = cfg["detect_neighbours"]
    seen, margins = [], []
    frames = file_frames(final, clip.width, clip.height, clip.n_frames)
    for k, bg in enumerate(clip.frames()):
        fin = block_mean(next(frames), f)[roi] / 255.0
        bgs = block_mean(bg, f) / 255.0
        cands = [-1] + [i for i, ly in enumerate(lays) if ly.first - K <= k < ly.last + K]
        errs = [float(np.abs(fin - predict(bgs, None if i < 0 else small[i], k, style, fps)[roi]).mean()) for i in cands]
        order = np.argsort(errs)
        seen.append(cands[order[0]])
        margins.append(errs[order[1]] / max(errs[order[0]], 1e-9) if len(errs) > 1 else float("inf"))
    next(frames, None)
    expected = [-1 if (ly := visible(lays, k)) is None else lays.index(ly) for k in range(clip.n_frames)]
    return seen, margins, expected, lays


# ------------------------------------------------------------------------------------- audio
def shutter_offsets(final, effects, shutter, frames, fps, sr, cfg):
    """Per caption frame, located in the FINAL audio by cross-correlation with the shutter-only track:
    the offset (ms) of the shutter's biggest wave from the frame, the lag (samples) and the shutter's level
    in the final relative to the plan (1.0 = all there; measured on the effects part, final minus voice)."""
    w, L = int(cfg["shutter_window_s"] * sr), int(0.02 * sr)
    out = []
    for fr in frames:
        i = int(round(fr / fps * sr))
        a, b = max(0, i - w), i + w
        sh, fin = shutter[a:b], final[a:b]
        n = len(sh) + len(fin)
        xc = np.fft.irfft(np.fft.rfft(fin, n) * np.conj(np.fft.rfft(sh, n)), n)
        lags = np.r_[np.arange(0, L + 1), np.arange(-L, 0)]
        lag = int(lags[int(np.argmax(np.r_[xc[:L + 1], xc[n - L:]]))])
        moved = shutter[max(0, a - lag):b - lag]                 # the shutter where the final has it
        fx = effects[b - len(moved):b] if a - lag < 0 else effects[a:a + len(moved)]
        level = float(np.dot(fx, moved) / (np.dot(moved, moved) + 1e-12))
        p = a + int(np.argmax(np.abs(sh)))                       # biggest wave as placed in the shutter track
        out.append(dict(offset_ms=round((p + lag - i) / sr * 1000, 3), lag_samples=lag,
                        track_offset_samples=p - i, level=round(level, 3)))
    return out


def onsets(effects, sr, cfg):
    """Times (s) where a sound starts in the effects part of the final audio: the level (RMS over a
    trailing window) rises by onset_rise_db over what it was shortly before, above a floor relative to
    the loudest moment. A smoothed level, so the hit's sub-bass cycles do not count as new sounds."""
    hop, win = int(cfg["onset_hop_s"] * sr), int(cfg["onset_rms_s"] * sr)
    c = np.concatenate([[0.0], np.cumsum(effects.astype(float) ** 2)])
    ends = np.arange(win, len(effects) + 1, hop)
    env = 10 * np.log10((c[ends] - c[ends - win]) / win + 1e-12)
    b0, b1 = (int(round(x / cfg["onset_hop_s"])) for x in cfg["onset_before_s"])
    rising = np.zeros(len(env), bool)
    for j in range(b1, len(env)):
        rising[j] = env[j] >= env.max() + cfg["onset_floor_db"] and env[j] - env[j - b1:j - b0 + 1].min() >= cfg["onset_rise_db"]
    starts = np.nonzero(rising & ~np.r_[False, rising[:-1]])[0]
    return [round(float(ends[j] / sr), 4) for j in starts]


# ------------------------------------------------------------------------------------- proofs
def plot_shutter_proof(final48, shutter48, caps, sr, fps, path):
    cols = min(6, len(caps))
    rows = math.ceil(len(caps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows + 0.9), sharey=True, squeeze=False)
    fig.suptitle("FINAL file: the shutter's biggest wave vs. the first frame of each caption", x=0.02, ha="left",
                 fontsize=14, color=INK, weight="bold")
    lim = max(np.abs(final48).max(), 1e-6)
    for ax, c in zip(axes.ravel(), caps):
        i = int(round(c["detected_first_frame"] / fps * sr))
        a, b = max(0, i - int(0.12 * sr)), i + int(0.12 * sr)
        t = (np.arange(a, a + len(final48[a:b])) - i) / sr * 1000
        ax.plot(t, final48[a:b], color=MUTED, lw=0.5, alpha=0.6, label="final audio")
        lag = c["shutter_lag_samples"]
        sh = shutter48[max(0, a - lag):b - lag]
        ax.plot(t[-len(sh):] if a - lag < 0 else t[:len(sh)], sh, color=BLUE, lw=0.8, label="shutter, as found in it")
        ax.axvline(0, color=ORANGE, lw=1.8)
        pk = c["shutter_offset_ms"]
        ax.plot([pk], [shutter48[i + c["shutter_track_offset_samples"]]], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2, zorder=5)
        ax.set_title(f'{c["text"]}\nbiggest wave {pk:+.1f} ms', fontsize=10, color=INK)
        ax.set_xlim(-120, 120)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("ms from caption frame")
    for ax in axes.ravel()[len(caps):]:
        ax.set_visible(False)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", fontsize=9, frameon=False, ncol=2)
    fig.text(0.02, 0.01, "orange line = first frame the caption is visible in final.mp4;  dot = the shutter's biggest wave "
             "located in the final audio by cross-correlation with the shutter-only track", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_timeline(final, effects, caps, words, starts, sr, fps, duration, path):
    fig, ax = plt.subplots(2, 1, figsize=(16, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    fig.suptitle("FINAL file: captions on screen, spoken words and sounds on one timeline", x=0.04, ha="left",
                 fontsize=15, color=INK, weight="bold")
    t = np.arange(len(final)) / sr
    ax[0].plot(t, final, color=MUTED, lw=0.4, label="final audio (voice + effects)")
    ax[0].plot(t[:len(effects)], effects, color=BLUE, lw=0.5, label="effects in the final audio (final minus voice)")
    top = np.abs(final).max()
    ax[0].plot(starts, [top * 1.05] * len(starts), "v", ms=5, color=BLUE, label="a sound starts")
    ax[0].set_ylabel("audio")
    ax[0].legend(loc="lower right", fontsize=9, frameon=False, ncol=3)
    for k, c in enumerate(caps):
        ft = c["detected_first_frame"] / fps
        end = caps[k + 1]["detected_first_frame"] / fps if k + 1 < len(caps) else duration
        for axx in ax:
            axx.axvline(ft, color=ORANGE, lw=1.2, zorder=3)
        colour = PINK if c["fill"] == "pink" else (BLUE if c["class"] == "script" else INK2)
        ax[1].add_patch(plt.Rectangle((ft, 0.55), end - ft - 0.004, 0.35, color=colour, alpha=0.85, lw=0))
        ax[1].text(ft + 0.01, 0.725, c["text"], fontsize=9, color="white", va="center", clip_on=True)
    for k, w in enumerate(words):
        ax[1].add_patch(plt.Rectangle((w["start"], 0.12), max(0.01, w["end"] - w["start"]), 0.25, color=GREEN, alpha=0.75, lw=0))
        ax[1].text(w["start"], 0.05 if k % 2 else 0.43, w["word"], fontsize=8, color=INK, va="center")
    ax[1].set_yticks([0.25, 0.725], ["words (re-aligned\non final audio)", "caption on screen\n(detected in final frames)"])
    ax[1].set_ylim(0, 1)
    ax[1].grid(visible=False)
    ax[1].set_xlim(0, duration)
    ax[1].set_xlabel("time in final.mp4 (s)")
    fig.text(0.04, 0.01, "orange line = frame a caption appears (each carries the shutter);  blue triangles = where a sound "
             "starts in the effects (each must be at a caption frame: the shutter's clicks and the hit)", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------- main
def verify(final, clip, plan, style, settings, work, out_dir, voice_wav, shutter_only_wav, mix_info, align_model):
    cfg = settings["verify"]
    fps, n = clip.fps, clip.n_frames
    sr = style["mix"]["sample_rate"]
    chk = Checks()

    # container and durations
    info = probe(final)
    tags = {info["color_space"], info["color_primaries"], info["color_transfer"]}
    chk.add("video format", info["width"] == clip.width and info["height"] == clip.height and abs(info["fps"] - fps) < 1e-6
            and info["frames"] == n and tags == {style["render"]["colour"]},
            size=f"{info['width']}x{info['height']}", fps=info["fps"], frames=info["frames"], expected_frames=n,
            colour=sorted(tags))
    mono = decode_audio(final, sr, channels=2).mean(axis=1)          # level-exact: average, not ffmpeg's downmix
    video_s, audio_s = n / fps, len(mono) / sr
    chk.add("audio and video durations match", abs(video_s - audio_s) <= cfg["av_tolerance_frames"] / fps,
            video_s=round(video_s, 4), audio_s=round(audio_s, 4), diff_frames=round((audio_s - video_s) * fps, 3))

    # loudness
    loud = loudness(final)
    target = style["mix"]["final_lufs"]
    chk.add("integrated loudness", abs(loud["lufs"] - target) <= cfg["lufs_tolerance"], lufs=loud["lufs"],
            target=target, tolerance=cfg["lufs_tolerance"])
    chk.add("peak below 0 dBFS", loud["sample_peak_dbfs"] < cfg["max_peak_dbfs"],
            sample_peak_dbfs=loud["sample_peak_dbfs"], true_peak_dbtp=loud["true_peak_dbtp"])

    # captions on screen, from the final frames
    seen, margins, expected, lays = detect_captions(final, clip, plan, style, cfg)
    wrong = [k for k in range(n) if seen[k] != expected[k]]
    changes = [k for k in range(n) if seen[k] != (seen[k - 1] if k else -1)]
    detected_first = [next((k for k in range(n) if seen[k] == i), None) for i in range(len(lays))]
    chk.add("caption changes in the final frames are the planned frames", not wrong,
            changes=len(changes), planned=len(lays), wrong_frames=wrong[:20],
            min_margin=round(min(margins[ly.first] for ly in lays), 2))

    # re-align the final audio: every caption 0-2 frames before its first word
    words, firsts = spoken_words(plan)
    realigned = align(decode_audio(final, 16000), [w["word"] for w in words], align_model,
                      settings["align"]["chunk_s"], settings["align"]["phases"])
    lo, hi = cfg["caption_lead_frames"]
    tol = cfg["align_tolerance_s"]
    caps = []
    for i, (c, ly) in enumerate(zip(plan["captions"], lays)):
        f = detected_first[i] if detected_first[i] is not None else ly.first
        w = realigned[firsts[i]]
        caps.append(dict(index=i + 1, text=c["text"], **{"class": c["class"]}, fill=c["fill"], planned_first_frame=ly.first,
                         detected_first_frame=f, detect_margin=round(margins[f], 2), first_word=w["word"],
                         word_start_final=w["start"], lead_ms=round((w["start"] - f / fps) * 1000, 1),
                         lead_frames=round((w["start"] - f / fps) * fps, 2), hit=bool(c["sfx"]["hit"])))
    bad = [c["text"] for c in caps if not (lo / fps - tol <= c["lead_ms"] / 1000 <= hi / fps + tol)]
    chk.add(f"every caption appears {lo}-{hi} frames before its first word (re-aligned final audio)", not bad,
            leads_frames=[c["lead_frames"] for c in caps], late_or_early=bad)

    # the voice alone through the identical chain and limiter; the rest of the final audio is the effects
    mix.voice_alone(voice_wav, f"{work}/voice_alone.wav", style["mix"], mix_info["voice_gain_db"], clip.duration)
    voice = decode_audio(f"{work}/voice_alone.wav", sr, channels=2).mean(axis=1)
    m = min(len(mono), len(voice))
    effects = mono[:m] - voice[:m]
    shutter = decode_audio(shutter_only_wav, sr, channels=2).mean(axis=1)[:m]

    # the shutter's biggest wave on every caption frame
    offs = shutter_offsets(mono[:m], effects, shutter, [c["detected_first_frame"] for c in caps], fps, sr, cfg)
    for c, o in zip(caps, offs):
        c.update(shutter_offset_ms=o["offset_ms"], shutter_lag_samples=o["lag_samples"],
                 shutter_track_offset_samples=o["track_offset_samples"], shutter_level=o["level"])
    worst = max(abs(o["offset_ms"]) for o in offs)
    lv0, lv1 = cfg["shutter_level_range"]
    missing = [c["text"] for c in caps if not lv0 <= c["shutter_level"] <= lv1]
    chk.add("shutter's biggest wave on every caption frame (final audio)", worst <= cfg["shutter_tolerance_ms"] and not missing,
            captions=len(caps), worst_offset_ms=worst, tolerance_ms=cfg["shutter_tolerance_ms"],
            levels=[c["shutter_level"] for c in caps], missing=missing)

    # no other sounds: every onset in the effects part sits at a caption frame
    starts = onsets(effects, sr, cfg)
    w0, w1 = cfg["allowed_onset_s"]
    frame_ts = [c["detected_first_frame"] / fps for c in caps]
    stray = [t for t in starts if not any(ft + w0 <= t <= ft + w1 for ft in frame_ts)]
    silent = [c["text"] for c, ft in zip(caps, frame_ts) if not any(ft + w0 <= t <= ft + w1 for t in starts)]
    track = decode_audio(f"{work}/sfx.wav", sr, channels=2).mean(axis=1)[:m]
    match = float(np.corrcoef(effects[:len(track)], track)[0, 1])
    chk.add("no sounds except at caption frames", not stray and not silent and match >= cfg["effects_min_correlation"],
            onsets=len(starts), stray_onsets_s=stray, captions_without_onset=silent,
            effects_vs_track_correlation=round(match, 4))

    # proofs
    plot_shutter_proof(mono, shutter, caps, sr, fps, f"{out_dir}/proof_shutter_alignment.png")
    plot_timeline(mono, effects, caps, realigned, starts, sr, fps, video_s, f"{out_dir}/proof_timeline.png")

    return dict(ok=chk.ok, final=final, checks=chk.items, captions=caps, loudness=loud,
                durations=dict(video_s=round(video_s, 6), audio_s=round(audio_s, 6), frames=n, fps=fps),
                caption_changes_detected=changes, sound_onsets_s=starts, words_final=realigned, mix=mix_info,
                proofs=[f"{out_dir}/proof_shutter_alignment.png", f"{out_dir}/proof_timeline.png"])
