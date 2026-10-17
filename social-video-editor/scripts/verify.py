"""Verify the FINISHED file against the plan and the client's rules; write report.json and proof images.

Everything is measured on final.mp4 itself (timing on its clean voice track, as the client asks):
  container   frame count, size, frame rate, bt709 tags; audio and video durations within a frame
  edges       the audio is near silence in its first and last edge_window_s: no click at the start or the end,
              and none where the video loops
  loudness    integrated loudness at the style's final target (+- tolerance), sample peak below 0 dBFS
  captions    which caption is on screen in every final frame: each frame is compared (at 1/detect_scale
              size, around the captions) with the clean clip frame composited with each nearby caption
              and with none; the closest wins. The caption changes found this way must be exactly the
              planned first frames. Captions that look identical (same text and colour, no zoom) are one
              "look": where one follows another with the same look there is nothing to see, so that
              caption's first frame is taken from the plan and reported as change_visible: false.
  timing      the clean voice track on the final timeline (the voice through the identical mix chain and
              limiter, with silent effects) is re-aligned to the transcript; every caption must appear 0-2
              frames before its first word (+- one alignment frame). Client rule: not the final mix, whose
              effects pull the aligner by up to ~100 ms (the mix alignment is reported for information).
  on screen   every caption is on screen for at least the style's grouping.min_frames frames (from the
              caption changes found in the final frames)
  placement   client rule: always centred. The style must ask for it (placement centre on the ink box at
              x = y = 0.5), each caption layer's ink box (every pixel it inks, the faintest hairline of a swash
              included) must be centred on the frame (within half a pixel: whole-pixel placement), and in
              every captioned final frame that layer must be where the final shows it: of the layer composited
              on the clean frame like the final (zoom included) at every shift within +-placement_search_px,
              the one that fits the final frame best must be within centre_tolerance_px of no shift. (Not
              the box of the pixels that visibly differ: the thin, low-alpha hairlines of a lopsided script
              word vanish in that, and a centred word then measured up to 7.5 px off.)
  blend       of the pixels that visibly differ from the clean frame (zoomed like the final), most match the
              style's blend (difference: |C - B|) and more of them than match a normal blend (C), per caption
              (client rule: the letters react to what moves behind)
  shutter     per caption, its own shutter alone (rebuilt at its planned pitch, level and pan with its biggest
              wave on the caption's first frame) is cross-correlated with the effects part of the final, both
              high-passed; the best lag (searched over +- shutter_max_lag_s, far less than the spacing of two
              captions, so a neighbour's shutter is never measured instead) is how far the biggest wave sits
              from the frame and must be within shutter_tolerance_ms; the match must be clear and the
              shutter in the final at about its planned level. On a script caption the hit starts on the same
              frame, so the loudest sample of the two together can be the hit's (reported as
              loudest_with_hit_ms); the shutter itself is still exactly on the frame.
  complete    every shutter plays in full: its first click (first sample within trim.first_click_rel_db of
              its biggest wave), where the shutter is found in the final, lies inside the video. A first
              caption on frame 0 or 1 would cut it off; edit_video.py moves the trim start back, or refuses
              to render, so that this cannot happen
  no extras   the final audio minus the voice (rebuilt through the identical chain and limiter with silent
              effects) must be the effects track (correlation); in that track (with silence before t = 0, so
              a sound at the very start is found too) every sound onset must sit at a caption frame (the
              shutter's clicks and the hit), every caption frame must have one, and between caption sounds
              the level may only fade: no whooshes, run-ups or other sounds
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import mix
import sfx
from align import align, shifted
from common import decode_audio, file_frames, loudness, probe
from plan import pauses_not_cut, spoken_words
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
    out[sl["y0"]:sl["y0"] + h, sl["x0"]:sl["x0"] + w] = reg * (1 - a) + sl["layer"].blend(reg, sl["layer"].colour) * a
    s = zoom_at(sl["layer"], k / fps, style["zoom"])
    if s > 1.0 + style["zoom"]["min_visible"]:
        out = apply_zoom((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), s, style["zoom"]["origin"]) / 255.0
    return out


def ink_blend(fin, ref, layer, cfg):
    """How the caption is blended in a finished frame, measured against the clean frame `ref` (zoomed like the
    final). Ink = pixels that differ from ref by more than ink_threshold in any channel (codec noise stays
    well below). Returns the share of ink pixels that match the layer's blend and a normal blend within
    blend_tolerance; None if nothing differs."""
    ink = np.abs(fin.astype(np.int16) - ref).max(axis=2) > cfg["ink_threshold"]
    if not ink.any():
        return None
    f, r = fin[ink].astype(np.float32), ref[ink].astype(np.float32) / 255.0
    blended = np.abs(f - layer.blend(r, layer.colour) * 255).max(axis=1)
    normal = np.abs(f - layer.colour * 255).max(axis=1)
    return dict(blend_match=float((blended < cfg["blend_tolerance"]).mean()),
                normal_match=float((normal < cfg["blend_tolerance"]).mean()), pixels=int(ink.sum()))


def layer_box_offset(layer, width, height):
    """Offset (px) of the centre of the layer's ink box (every pixel its alpha touches, the faintest hairline
    included: the typesetter crops the alpha to exactly that) from the frame centre."""
    h, w = layer.alpha.shape
    return layer.x0 + w / 2 - width / 2, layer.y0 + h / 2 - height / 2


def zoomed_alpha(layer, s, origin, width, height):
    """The layer's alpha as it lands in the final frame under zoom s: the same bicubic resampling about the zoom
    origin as render_captions.apply_zoom, computed for the layer's own box only. Returns (x0, y0, alpha) in
    final-frame pixels (the alpha unchanged when there is no zoom)."""
    h, w = layer.alpha.shape
    if s == 1.0:
        return layer.x0, layer.y0, layer.alpha
    m = 4                                                    # zero margin, wider than the bicubic support
    canvas = np.zeros((h + 2 * m, w + 2 * m), np.float32)
    canvas[m:m + h, m:m + w] = layer.alpha
    bx0, by0 = origin[0] * width - width / (2 * s), origin[1] * height - height / (2 * s)   # apply_zoom's box
    X0, X1 = math.floor((layer.x0 - 2 - bx0) * s), math.ceil((layer.x0 + w + 2 - bx0) * s)
    Y0, Y1 = math.floor((layer.y0 - 2 - by0) * s), math.ceil((layer.y0 + h + 2 - by0) * s)
    cx0, cy0 = layer.x0 - m, layer.y0 - m
    img = Image.fromarray(canvas, "F").resize((X1 - X0, Y1 - Y0), Image.BICUBIC,
                                              box=(bx0 + X0 / s - cx0, by0 + Y0 / s - cy0, bx0 + X1 / s - cx0, by0 + Y1 / s - cy0))
    return X0, Y0, np.asarray(img, np.float32)


def framed(img, y0, x0, h, w):
    """img[y0:y0 + h, x0:x0 + w] as float 0..1 with zeros outside the frame, and the mask of the pixels inside."""
    H, W = img.shape[:2]
    out = np.zeros((h, w) + img.shape[2:], np.float32)
    mask = np.zeros((h, w), bool)
    ys, ye, xs, xe = max(0, y0), min(H, y0 + h), max(0, x0), min(W, x0 + w)
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe] / 255.0
        mask[ys - y0:ye - y0, xs - x0:xe - x0] = True
    return out, mask


def correlate_shifts(G, A, R):
    """c[R + dy, R + dx] = sum over the layer of A[i, j] * G[R + dy + i, R + dx + j], for |dx|, |dy| <= R
    (G is A's box grown by R on every side, so no shift wraps around)."""
    c = np.fft.irfft2(np.fft.rfft2(G) * np.conj(np.fft.rfft2(A, s=G.shape)), s=G.shape)
    return c[:2 * R + 1, :2 * R + 1]


def placement(fin, ref, layer, s, zcfg, R):
    """Where the caption layer sits in a finished frame, as the shift of the expected render that fits the frame
    best. The expected frame for a shift (dx, dy) is the clean frame zoomed like the final (ref) with the
    layer's alpha, zoomed the same way (zoomed_alpha) and moved by (dx, dy), composited in its blend:
    ref + A * D with D = blend(ref, C) - ref. Its squared error against the final frame over the layer's box
    grown by R, E(dx, dy) = E_none + 2 corr(A, D.(ref - fin)) + corr(A^2, D.D), is computed for every shift
    within +-R px at once (pixels outside the frame left out). The final zooms the composite, this composites
    on the zoomed frame: the two differ by under 1% of the codec's own error (Tears of Steel, script words at
    full zoom), and synthetic 1-3 px offsets are found exactly with and without zoom. Returns the best shift
    (final-frame px) and fit = E_none / E_best, how much better the caption there explains the frame than no
    caption (about 1 where the caption cannot be seen, e.g. white over mid grey in a difference blend)."""
    H, W = fin.shape[:2]
    X0, Y0, A = zoomed_alpha(layer, s, zcfg["origin"], W, H)
    h, w = A.shape
    F, mask = framed(fin, Y0 - R, X0 - R, h + 2 * R, w + 2 * R)
    B, _ = framed(ref, Y0 - R, X0 - R, h + 2 * R, w + 2 * R)
    D = layer.blend(B, layer.colour) - B
    P = B - F
    A = A.astype(np.float64)
    E = float((P * P).sum()) + 2 * correlate_shifts((D * P).sum(axis=2) * mask, A, R) \
        + correlate_shifts((D * D).sum(axis=2) * mask, A * A, R)
    j = np.unravel_index(int(np.argmin(E)), E.shape)
    e_none = float((P * P).sum())
    return dict(dx=int(j[1]) - R, dy=int(j[0]) - R, fit=e_none / max(float(E[j]), 1e-12))


def looks(lays):
    """Per caption, the index of the first caption that renders identically (same text and colour, no zoom:
    a zoomed caption changes with its own start time, so it is always its own look)."""
    first = {}
    return [i if ly.zoom else first.setdefault((ly.text, tuple(ly.colour), ly.x0, ly.y0), i) for i, ly in enumerate(lays)]


def detect_captions(final, clip, plan, style, cfg, scaler):
    """For every final frame: the look (see looks()) on screen (-1 = none) and the decision margin between the
    best and the second-best look, and for frames with a caption its blend (ink_blend) and placement
    (placement), both against the clean frame zoomed for the caption found; the expected look per frame; the
    layers and their looks."""
    f, fps = cfg["detect_scale"], clip.fps
    lays = layers(plan, style, clip.width, clip.height)
    look = looks(lays)
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
    seen, margins, geometry = [], [], []
    frames = file_frames(final, clip.width, clip.height, clip.n_frames)
    for k, bg in enumerate(clip.frames(scaler)):
        full = next(frames)
        fin = block_mean(full, f)[roi] / 255.0
        bgs = block_mean(bg, f) / 255.0
        near = {}                                                # one caption per look
        for i, ly in enumerate(lays):
            if ly.first - K <= k < ly.last + K:
                near.setdefault(look[i], i)
        cands = [-1] + list(near.values())
        errs = [float(np.abs(fin - predict(bgs, None if i < 0 else small[i], k, style, fps)[roi]).mean()) for i in cands]
        order = np.argsort(errs)
        best = cands[order[0]]
        seen.append(-1 if best < 0 else look[best])
        margins.append(errs[order[1]] / max(errs[order[0]], 1e-9) if len(errs) > 1 else float("inf"))
        geo = None
        if best >= 0:
            s = zoom_at(lays[best], k / fps, style["zoom"])
            s = s if s > 1.0 + style["zoom"]["min_visible"] else 1.0          # as render_captions.finished_frame
            ref = apply_zoom(bg, s, style["zoom"]["origin"]) if s > 1.0 else bg
            geo = dict(caption=best, blend=ink_blend(full, ref, lays[best], cfg),
                       place=placement(full, ref, lays[best], s, style["zoom"], cfg["placement_search_px"]))
        geometry.append(geo)
    next(frames, None)
    expected = [-1 if (ly := visible(lays, k)) is None else look[lays.index(ly)] for k in range(clip.n_frames)]
    return seen, margins, geometry, expected, lays, look


# ------------------------------------------------------------------------------------- audio
def highpass_fir(cut_hz, sr, taps=511):
    """Linear-phase windowed-sinc high-pass with an odd number of taps (a whole-sample delay of taps // 2)."""
    n = np.arange(taps) - taps // 2
    lp = np.sinc(2 * cut_hz / sr * n) * np.blackman(taps)
    hp = -lp / lp.sum()
    hp[taps // 2] += 1
    return hp


def window(x, a, b):
    """x[a:b], with zeros where the range runs off either end."""
    out = np.zeros(b - a)
    lo, hi = max(a, 0), min(b, len(x))
    if hi > lo:
        out[lo - a:hi - a] = x[lo:hi]
    return out


def shutter_offsets(effects, captions, frames, style, fps, sr, cfg, click_rel_db):
    """Where each caption's shutter is in the FINAL audio. effects: the effects part of the final (mono).
    Each caption's own shutter is rebuilt alone (planned pitch, level and pan) with its biggest wave on the
    caption's first frame, and slid against the effects over lags up to shutter_max_lag_s; both are
    high-passed (shutter_highpass_hz) so the hit's sub-bass under a script caption does not pull the match.
    Returns per caption: offset of the biggest wave from the frame (the best lag), the normalised correlation
    there, the shutter's level in the final relative to the plan (1.0 = all there, full band), and where its
    first click (sfx.onset: its first sample within click_rel_db of the biggest wave) is in the final (s;
    negative = cut off at the start of the video); and per caption the template as found (for the proof)."""
    base, anchor = sfx.load_shutter(style["sfx"]["shutter"], sr)
    L = int(round(cfg["shutter_max_lag_s"] * sr))
    fir = highpass_fir(cfg["shutter_highpass_hz"], sr)
    h = len(fir) // 2
    rows, found = [], []
    for f, c in zip(frames, captions):
        p = c["sfx"]["shutter"]
        snd, peak = sfx.repitch(base, p["pitch"], anchor)
        tmpl = snd * sfx.db(p["gain_db"]) * np.mean(sfx.pan_gains(p["pan"]))   # what it adds to the mono mix
        s0 = sfx.frame_sample(f, fps, sr) - peak                                # its start if exactly on the frame
        th = np.convolve(tmpl, fir)                                             # high-passed, from sample s0 - h
        seg = np.convolve(window(effects, s0 - 2 * h - L, s0 + len(tmpl) + 2 * h + L), fir, "valid")  # from s0 - h - L
        xc = np.correlate(seg, th, "valid")                                     # index L + lag
        j = int(np.argmax(xc))
        lag = j - L
        corr = xc[j] / (np.linalg.norm(th) * np.linalg.norm(seg[j:j + len(th)]) + 1e-12)
        level = np.dot(window(effects, s0 + lag, s0 + lag + len(tmpl)), tmpl) / np.dot(tmpl, tmpl)
        rows.append(dict(offset_ms=round(lag / sr * 1000, 3), lag_samples=lag, correlation=round(float(corr), 3),
                         level=round(float(level), 3), first_click_s=round((s0 + lag + sfx.onset(snd, click_rel_db)) / sr, 4)))
        found.append(dict(start=s0 + lag, sound=tmpl, peak=peak))
    return rows, found


def level_db(x, sr, win_s, hop_s, highpass_hz=0):
    """Level (dB) over a trailing window every hop: (times of the window ends, levels)."""
    if highpass_hz:
        X = np.fft.rfft(x)
        X[np.fft.rfftfreq(len(x), 1 / sr) < highpass_hz] = 0
        x = np.fft.irfft(X, len(x))
    hop, win = int(hop_s * sr), int(win_s * sr)
    c = np.concatenate([[0.0], np.cumsum(x.astype(float) ** 2)])
    ends = np.arange(win, len(x) + 1, hop)
    return ends / sr, 10 * np.log10((c[ends] - c[ends - win]) / win + 1e-12)


def onsets(track, sr, cfg):
    """Times (s) where a sound starts in the effects track: the level above onset_highpass_hz (so the hit's
    sub-bass cycles do not count) rises onset_rise_db over what it was shortly before, above a floor
    relative to the loudest moment. The track is measured with silence before t = 0 (the RMS window plus the
    look-back), so a sound that is already playing on the first sample is found too, rising from nothing."""
    pad = int(math.ceil((cfg["onset_rms_s"] + cfg["onset_before_s"][1]) / cfg["onset_hop_s"])) * int(cfg["onset_hop_s"] * sr)
    t, env = level_db(np.r_[np.zeros(pad), track], sr, cfg["onset_rms_s"], cfg["onset_hop_s"], cfg["onset_highpass_hz"])
    t = t - pad / sr
    b0, b1 = (int(round(x / cfg["onset_hop_s"])) for x in cfg["onset_before_s"])
    rising = np.zeros(len(env), bool)
    for j in range(b1, len(env)):
        rising[j] = env[j] >= env.max() + cfg["onset_floor_db"] and env[j] - env[j - b1:j - b0 + 1].min() >= cfg["onset_rise_db"]
    return [round(float(t[j]), 4) for j in np.nonzero(rising & ~np.r_[False, rising[:-1]])[0]]


def swells(track, sr, cfg, frame_ts):
    """Between caption sounds the effects may only fade out. Returns [(time, dB)] wherever the level rises
    more than fade_rise_db above its lowest point since the previous caption's sound (a run-up, whoosh or
    any other sound that is not a caption sound), above fade_floor_db re the loudest moment."""
    t, env = level_db(track, sr, cfg["fade_rms_s"], cfg["onset_hop_s"])
    w0, w1 = cfg["allowed_onset_s"]
    gaps = [(0.0 if k == 0 else frame_ts[k - 1] + w1, ft + w0) for k, ft in enumerate(frame_ts)] + [(frame_ts[-1] + w1, t[-1] + 1)]
    out = []
    for a, b in gaps:
        sel = (t > a) & (t < b)
        if sel.sum() < 2:
            continue
        lv = env[sel]
        rise = np.where(lv >= env.max() + cfg["fade_floor_db"], lv - np.minimum.accumulate(lv), -np.inf)
        k = int(np.argmax(rise))
        if rise[k] > cfg["fade_rise_db"]:
            out.append((round(float(t[sel][k]), 3), round(float(rise[k]), 1)))
    return out


# ------------------------------------------------------------------------------------- proofs
def plot_shutter_proof(final48, found, caps, sr, fps, path):
    with plt.rc_context({"path.simplify": False}):          # keep every sample: the biggest wave is one spike
        _plot_shutter_proof(final48, found, caps, sr, fps, path)


def frame_title(fig, title, note, height_in):
    """Title at the top and note at the bottom, a fixed distance in inches from the edges (any figure height);
    returns the tight_layout rect that leaves room for them."""
    fig.suptitle(title, x=0.02, y=1 - 0.3 / height_in, ha="left", va="top", fontsize=14, color=INK, weight="bold")
    fig.text(0.02, 0.15 / height_in, note, fontsize=9, color=INK2)
    return (0, 0.45 / height_in, 1, 1 - 0.75 / height_in)


def _plot_shutter_proof(final48, found, caps, sr, fps, path):
    cols = min(6, len(caps))
    rows = math.ceil(len(caps) / cols)
    height = 3.4 * rows + 1.2
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, height), sharey=True, squeeze=False)
    rect = frame_title(fig, "FINAL file: the shutter's biggest wave vs. the first frame of each caption",
                       "orange line = first frame the caption is visible in final.mp4;  dot = the shutter's biggest wave "
                       "(the file's loudest sample), located in the final audio by cross-correlating the caption's own "
                       "shutter", height)
    lim = max(np.abs(final48).max(), 1e-6)
    for ax, c, fd in zip(axes.ravel(), caps, found):
        i = sfx.frame_sample(c["detected_first_frame"], fps, sr)
        a, b = max(0, i - int(0.12 * sr)), i + int(0.12 * sr)
        t = (np.arange(a, a + len(final48[a:b])) - i) / sr * 1000
        ax.plot(t, final48[a:b], color=MUTED, lw=0.5, alpha=0.6, label="final audio")
        snd = fd["sound"]
        ax.plot((fd["start"] + np.arange(len(snd)) - i) / sr * 1000, snd, color=BLUE, lw=0.8, zorder=3,
                label="this caption's shutter, where it is found in the final")
        ax.axvline(0, color=ORANGE, lw=1.8, zorder=2)
        pk = c["shutter_offset_ms"]
        ax.plot([pk], [snd[fd["peak"]]], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2, zorder=5)
        ax.set_title(f'{c["text"]}{" (+ hit)" if c["hit"] else ""}\nbiggest wave {pk:+.1f} ms', fontsize=10, color=INK)
        ax.set_xlim(-120, 120)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("ms from caption frame")
    for ax in axes.ravel()[len(caps):]:
        ax.set_visible(False)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.99, 1 - 0.3 / height),
               fontsize=9, frameon=False, ncol=2)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_timeline(final, effects, caps, words, starts, sr, fps, duration, path, row_s=10.0, single_row_up_to_s=15.0):
    """One timeline of the final file; longer clips are split into stacked rows of row_s seconds."""
    n_rows = 1 if duration <= single_row_up_to_s else math.ceil(duration / row_s)
    span = duration / n_rows if n_rows == 1 else row_s
    height = 6.6 * n_rows + 1.2
    fig, axes = plt.subplots(2 * n_rows, 1, figsize=(16, height), squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1] * n_rows})
    rect = frame_title(fig, "FINAL file: captions on screen, spoken words and sounds on one timeline",
                       "orange line = frame a caption appears (each carries the shutter);  blue triangles = where a sound "
                       "starts in the effects (each must be at a caption frame: the shutter's clicks and the hit)", height)
    top = np.abs(final).max()
    ends = [c["detected_first_frame"] / fps for c in caps[1:]] + [duration]
    for r in range(n_rows):
        ax_a, ax_c = axes[2 * r][0], axes[2 * r + 1][0]
        t0, t1 = r * span, min(duration, (r + 1) * span)
        i0, i1 = int(t0 * sr), int(math.ceil(t1 * sr))
        t = np.arange(i0, i0 + len(final[i0:i1])) / sr
        ax_a.plot(t, final[i0:i1], color=MUTED, lw=0.4, label="final audio (voice + effects)")
        ax_a.plot(t[:len(effects[i0:i1])], effects[i0:i1], color=BLUE, lw=0.5,
                  label="effects in the final audio (final minus voice)")
        st = [x for x in starts if t0 <= x <= t1]
        ax_a.plot(st, [top * 1.05] * len(st), "v", ms=5, color=BLUE, label="a sound starts")
        ax_a.set_ylabel("audio")
        ax_a.set_ylim(-top * 1.15, top * 1.15)
        if r == 0:                                           # above the panel, never over the waveform
            leg = ax_a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=9, ncol=3, frameon=False,
                              borderaxespad=0)
            for line in leg.get_lines():
                line.set_linewidth(2)
        for c, end in zip(caps, ends):
            ft = c["detected_first_frame"] / fps
            if end < t0 or ft > t1:
                continue
            for axx in (ax_a, ax_c):
                axx.axvline(ft, color=ORANGE, lw=1.2, zorder=3)
            colour = PINK if c["fill"] == "pink" else (BLUE if c["class"] == "script" else INK2)
            ax_c.add_patch(plt.Rectangle((ft, 0.55), end - ft - 0.004, 0.35, color=colour, alpha=0.85, lw=0))
            ax_c.text(max(ft, t0) + 0.01, 0.725, c["text"], fontsize=9, color="white", va="center", clip_on=True)
        for k, w in enumerate(words):
            if w["end"] < t0 or w["start"] > t1:
                continue
            ax_c.add_patch(plt.Rectangle((w["start"], 0.12), max(0.01, w["end"] - w["start"]), 0.25, color=GREEN,
                                         alpha=0.75, lw=0))
            ax_c.text(w["start"], 0.05 if k % 2 else 0.43, w["word"], fontsize=8, color=INK, va="center", clip_on=True)
        ax_c.set_yticks([0.25, 0.725], ["words (clean voice,\nre-aligned)", "caption on screen\n(detected in final frames)"])
        ax_c.set_ylim(0, 1)
        ax_c.grid(visible=False)
        for axx in (ax_a, ax_c):
            axx.set_xlim(t0, t0 + span)
        ax_c.set_xlabel("time in final.mp4 (s)")
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------- main
def verify(final, clip, plan, style, settings, work, out_dir, voice_wav, mix_info, align_model):
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
    stereo = decode_audio(final, sr, channels=2)
    mono = stereo.mean(axis=1)                                        # level-exact: average, not ffmpeg's downmix
    video_s, audio_s = n / fps, len(mono) / sr
    chk.add("audio and video durations match", abs(video_s - audio_s) <= cfg["av_tolerance_frames"] / fps,
            video_s=round(video_s, 4), audio_s=round(audio_s, 4), diff_frames=round((audio_s - video_s) * fps, 3))

    # the audio starts and ends in silence (up to the video's last frame; AAC pads beyond it), so there is
    # no click at either edge or where the video loops
    end = min(len(stereo), int(round(video_s * sr)))
    w = max(1, int(round(cfg["edge_window_s"] * sr)))

    def dbfs(x):
        return round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 1)
    edges = dict(start_dbfs=dbfs(stereo[:w]), end_dbfs=dbfs(stereo[end - w:end]),
                 loop_step_dbfs=dbfs(stereo[end - 1] - stereo[0]),
                 first_10ms_dbfs=dbfs(stereo[:int(0.01 * sr)]), last_10ms_dbfs=dbfs(stereo[end - int(0.01 * sr):end]))
    chk.add(f"audio starts and ends in silence (first and last {cfg['edge_window_s'] * 1000:g} ms)",
            max(edges["start_dbfs"], edges["end_dbfs"]) < cfg["edge_max_dbfs"], max_dbfs=cfg["edge_max_dbfs"], **edges)

    # loudness
    loud = loudness(final)
    target = style["mix"]["final_lufs"]
    chk.add("integrated loudness", abs(loud["lufs"] - target) <= cfg["lufs_tolerance"], lufs=loud["lufs"],
            target=target, tolerance=cfg["lufs_tolerance"])
    chk.add("peak below 0 dBFS", loud["sample_peak_dbfs"] < cfg["max_peak_dbfs"],
            sample_peak_dbfs=loud["sample_peak_dbfs"], true_peak_dbtp=loud["true_peak_dbtp"])

    # captions on screen, from the final frames
    seen, margins, geometry, expected, lays, look = detect_captions(final, clip, plan, style, cfg, settings["output"]["scaler"])
    wrong = [k for k in range(n) if seen[k] != expected[k]]
    changes = [k for k in range(n) if seen[k] != (seen[k - 1] if k else -1)]
    planned_changes = [k for k in range(n) if expected[k] != (expected[k - 1] if k else -1)]
    # each caption's first frame as seen in the final: the change into its look nearest to its planned frame;
    # a caption that follows one with the same look has no visible change, so its planned frame is used
    visible_change = [i == 0 or look[i] != look[i - 1] for i in range(len(lays))]
    detected_first = [min((k for k in changes if seen[k] == look[i] and abs(k - ly.first) <= cfg["detect_neighbours"]),
                          key=lambda k: abs(k - ly.first), default=None) if visible_change[i] else ly.first
                      for i, ly in enumerate(lays)]
    chk.add("caption changes in the final frames are the planned frames", not wrong,
            changes=len(changes), planned=len(planned_changes), wrong_frames=wrong[:20],
            same_look_as_previous=[ly.text for ly, v in zip(lays, visible_change) if not v],
            min_margin=round(min(margins[ly.first] for ly, v in zip(lays, visible_change) if v), 2))
    firsts_seen = [ly.first if f is None else f for f, ly in zip(detected_first, lays)] + [n]
    on_screen = [firsts_seen[k + 1] - firsts_seen[k] for k in range(len(lays))]
    min_f = style["grouping"]["min_frames"]
    chk.add(f"every caption on screen for at least {min_f} frames (final frames)", min(on_screen) >= min_f,
            shortest_frames=min(on_screen), too_short=[(ly.text, s) for ly, s in zip(lays, on_screen) if s < min_f])

    # placement and blend, measured in the final frames of each caption (only frames where that caption's look
    # was found)
    geo = [[g for g in geometry[firsts_seen[i]:firsts_seen[i + 1]] if g and look[g["caption"]] == look[i]]
           for i in range(len(lays))]
    centre = []
    for ly, g in zip(lays, geo):
        placed = [x["place"] for x in g if x["place"]["fit"] >= cfg["placement_min_fit"]]
        blends = [x["blend"] for x in g if x["blend"]]
        centre.append(dict(frames_measured=len(placed),
                           layer_box_offset_px=[round(v, 1) for v in layer_box_offset(ly, clip.width, clip.height)],
                           worst_shift_px=[max(abs(p["dx"]) for p in placed), max(abs(p["dy"]) for p in placed)] if placed else None,
                           min_fit=round(min(p["fit"] for p in placed), 1) if placed else None,
                           blend_match=round(float(np.mean([x["blend_match"] for x in blends])), 2) if blends else None,
                           normal_match=round(float(np.mean([x["normal_match"] for x in blends])), 2) if blends else None))
    lay = style["layout"]
    asks_centre = (lay["placement"], lay["anchor"], lay["x"], lay["y"]) == ("centre", "ink_box_centre", 0.5, 0.5)
    tol_px, R = cfg["centre_tolerance_px"], cfg["placement_search_px"]
    off = [(ly.text, c["layer_box_offset_px"], c["worst_shift_px"]) for ly, c in zip(lays, centre)
           if not c["frames_measured"] or max(map(abs, c["layer_box_offset_px"])) > 0.5 or max(c["worst_shift_px"]) > tol_px]
    measured = [c for c in centre if c["frames_measured"]]
    chk.add(f"every caption centred in the frame (layer ink box centred, best-fit shift in the final frames within {tol_px} px)",
            asks_centre and not off, style_layout=f"{lay['placement']}, {lay['anchor']} at ({lay['x']}, {lay['y']})",
            frames_measured=sum(c["frames_measured"] for c in centre), search_px=R,
            worst_dx_px=max((c["worst_shift_px"][0] for c in measured), default=None),
            worst_dy_px=max((c["worst_shift_px"][1] for c in measured), default=None),
            worst_layer_box_offset_px=max(abs(v) for c in centre for v in c["layer_box_offset_px"]),
            min_fit=min((c["min_fit"] for c in measured), default=None), off_centre_or_not_found=off)
    mode = style["blend"]["mode"]
    blended = [c for c in centre if c["blend_match"] is not None]
    not_blended = [ly.text for ly, c in zip(lays, centre) if c["blend_match"] is None
                   or c["blend_match"] < cfg["blend_min_match"] or c["blend_match"] <= c["normal_match"]]
    chk.add(f"captions drawn in a {mode} blend (final frames)", not not_blended,
            min_blend_match=min((c["blend_match"] for c in blended), default=None),
            max_normal_match=max((c["normal_match"] for c in blended), default=None),
            required=cfg["blend_min_match"], not_blended=not_blended)

    # the voice alone through the identical chain and limiter (silent effects): the clean voice track on the
    # final timeline. The rest of the final audio is the effects.
    mix.voice_alone(voice_wav, f"{work}/voice_alone.wav", style["mix"], mix_info["voice_gain_db"], clip.duration,
                    clip.voice_start_s)
    voice = decode_audio(f"{work}/voice_alone.wav", sr, channels=2).mean(axis=1)
    m = min(len(mono), len(voice))
    effects = mono[:m] - voice[:m]

    # timing: re-align the CLEAN voice track on the final timeline (client rule: effects in the mix pull the
    # aligner by up to ~100 ms); every caption must appear 0-2 frames before its first word. The alignment
    # of the final mix itself is reported alongside for information.
    words, firsts = spoken_words(plan)
    al = dict(model_name=align_model, chunk_s=settings["align"]["chunk_s"], phases=settings["align"]["phases"])
    # both aligned from where the voice starts (silent before clip.voice_start_s when the start was moved back
    # for the first caption's shutter: the timings then do not depend on the move)
    v0 = int(round(clip.voice_start_s * 16000))
    texts = [w["word"] for w in words]
    realigned = shifted(align(decode_audio(f"{work}/voice_alone.wav", 16000)[v0:], texts, **al), clip.voice_start_s)
    in_mix = shifted(align(decode_audio(final, 16000)[v0:], texts, **al), clip.voice_start_s)
    lo, hi = cfg["caption_lead_frames"]
    tol = cfg["align_tolerance_s"]
    caps = []
    for i, (c, ly) in enumerate(zip(plan["captions"], lays)):
        f = detected_first[i] if detected_first[i] is not None else ly.first
        w = realigned[firsts[i]]
        caps.append(dict(index=i + 1, text=c["text"], **{"class": c["class"]}, fill=c["fill"], planned_first_frame=ly.first,
                         detected_first_frame=f, change_visible=visible_change[i], frames_on_screen=on_screen[i],
                         placement_worst_shift_px=centre[i]["worst_shift_px"], placement_min_fit=centre[i]["min_fit"],
                         layer_box_offset_px=centre[i]["layer_box_offset_px"],
                         blend_match=centre[i]["blend_match"], normal_blend_match=centre[i]["normal_match"],
                         detect_margin=round(margins[f], 2),
                         first_word=w["word"],
                         word_start_final=w["start"], lead_ms=round((w["start"] - f / fps) * 1000, 1),
                         lead_frames=round((w["start"] - f / fps) * fps, 2),
                         lead_frames_in_mix=round((in_mix[firsts[i]]["start"] - f / fps) * fps, 2), hit=bool(c["sfx"]["hit"])))
    bad = [c["text"] for c in caps if not (lo / fps - tol <= c["lead_ms"] / 1000 <= hi / fps + tol)]
    chk.add(f"every caption appears {lo}-{hi} frames before its first word (clean voice re-aligned on the final timeline)",
            not bad, leads_frames=[c["lead_frames"] for c in caps], late_or_early=bad,
            for_information_leads_in_final_mix=[c["lead_frames_in_mix"] for c in caps])

    # the shutter's biggest wave on every caption frame
    click_db = settings["trim"]["first_click_rel_db"]
    offs, found = shutter_offsets(effects, plan["captions"], [c["detected_first_frame"] for c in caps], style, fps, sr, cfg,
                                  click_db)
    for c, o, fd in zip(caps, offs, found):
        c.update(shutter_offset_ms=o["offset_ms"], shutter_lag_samples=o["lag_samples"],
                 shutter_correlation=o["correlation"], shutter_level=o["level"], shutter_first_click_s=o["first_click_s"])
        if c["hit"]:
            # for information: the loudest sample of shutter + hit together, over the shutter's span
            a = max(0, fd["start"])
            j = a + int(np.argmax(np.abs(effects[a:fd["start"] + len(fd["sound"])])))
            c["loudest_with_hit_ms"] = round((j - sfx.frame_sample(c["detected_first_frame"], fps, sr)) / sr * 1000, 3)
    worst = max(abs(o["offset_ms"]) for o in offs)
    lv0, lv1 = cfg["shutter_level_range"]
    unclear = [c["text"] for c in caps if c["shutter_correlation"] < cfg["shutter_min_correlation"]
               or not lv0 <= c["shutter_level"] <= lv1]
    chk.add("shutter's biggest wave on every caption frame (final audio)", worst <= cfg["shutter_tolerance_ms"] and not unclear,
            captions=len(caps), worst_offset_ms=worst, tolerance_ms=cfg["shutter_tolerance_ms"],
            min_correlation=min(c["shutter_correlation"] for c in caps), levels=[c["shutter_level"] for c in caps],
            unclear_or_wrong_level=unclear)
    # every shutter in full: a first caption on frame 0 or 1 would cut its first clicks off at the start
    cut = [dict(caption=c["index"], text=c["text"], frame=c["detected_first_frame"],
                first_click_ms=round(c["shutter_first_click_s"] * 1000, 1)) for c in caps if c["shutter_first_click_s"] < 0]
    chk.add("every shutter complete (no clicks cut off at the start of the video)", not cut,
            first_shutter_first_click_ms=round(caps[0]["shutter_first_click_s"] * 1000, 1), first_click_rel_db=click_db,
            cut_off="; ".join(f"caption {x['caption']} ({x['text']!r}, frame {x['frame']}): its first click would be at "
                              f"{x['first_click_ms']} ms, before the video starts" for x in cut) or "none")

    # no other sounds: the effects part of the final IS the effects track, and in that track every sound
    # starts at a caption frame and only fades in between (measured on the clean track: no codec noise or
    # limiter interplay with the voice)
    track = decode_audio(f"{work}/sfx.wav", sr, channels=2).mean(axis=1)[:m]
    match = float(np.corrcoef(effects[:len(track)], track)[0, 1])
    starts = onsets(track, sr, cfg)
    w0, w1 = cfg["allowed_onset_s"]
    frame_ts = [c["detected_first_frame"] / fps for c in caps]
    stray = [t for t in starts if not any(ft + w0 <= t <= ft + w1 for ft in frame_ts)]
    silent = [c["text"] for c, ft in zip(caps, frame_ts) if not any(ft + w0 <= t <= ft + w1 for t in starts)]
    cut_texts = {x["text"] for x in cut}
    rising = swells(track, sr, cfg, frame_ts)
    chk.add("no sounds except at caption frames", match >= cfg["effects_min_correlation"] and not stray and not silent
            and not rising, effects_vs_track_correlation=round(match, 4), onsets=len(starts), stray_onsets_s=stray,
            captions_without_onset=[t + " (its shutter is cut off at the start of the video)" if t in cut_texts else t
                                    for t in silent], rising_between_captions=rising)

    # PREFERENCES.md 'Editing' (cut pauses > ~0.3 s) is not applied to Prime v11: report what it would cut,
    # measured on the clean voice of the final timeline
    pauses = pauses_not_cut(realigned, settings["editing"])
    warnings = [f"PREFERENCES.md 'Editing': {len(pauses)} pause(s) longer than {settings['editing']['pause_s']} s "
                f"{settings['editing']['pause_status']}: " + ", ".join(f"{p['after']!r} -> {p['before']!r} {p['length_s']:.2f} s"
                                                                        for p in pauses)] if pauses else []
    tail = video_s - realigned[-1]["end"]
    if tail < settings["trim"]["tail_s"] - 0.5 / fps:
        warnings.append(f"the clip ends {tail:.2f} s after the last word (trim.tail_s {settings['trim']['tail_s']} s): the "
                        "recording (or --end) stops there. A phone's stop-tap in the last frames is faded out with the "
                        f"voice over its last {style['mix']['voice_fade']['out_s'] * 1000:g} ms; if one is still audible, "
                        "end earlier with --end.")
    if clip.tonemap:
        warnings.append(f"HDR source ({clip.tonemap['transfer']}) tone-mapped to bt709 SDR (Mobius); colours can differ "
                        "from an SDR recording. For the approved look record SDR (iPhone: Settings > Camera > Record "
                        "Video > HDR Video off).")

    # proofs
    plot_shutter_proof(mono, found, caps, sr, fps, f"{out_dir}/proof_shutter_alignment.png")
    plot_timeline(mono, effects, caps, realigned, starts, sr, fps, video_s, f"{out_dir}/proof_timeline.png")

    notes = [f"Script captions ({', '.join(repr(c['text']) for c in caps if c['hit'])}) play the shutter and the hit "
             "together, the hit starting on the same frame: the loudest sample of the two combined (loudest_with_hit_ms) "
             "includes the hit and can sit off the frame. The shutter itself (located alone, by cross-correlation: "
             "shutter_offset_ms) has its biggest wave exactly on the caption frame."] if any(c["hit"] for c in caps) else []
    return dict(ok=chk.ok, final=final, checks=chk.items, warnings=warnings, notes=notes, captions=caps, loudness=loud,
                durations=dict(video_s=round(video_s, 6), audio_s=round(audio_s, 6), frames=n, fps=fps),
                caption_changes_detected=changes, sound_onsets_s=starts, audio_edges=edges, pauses_not_cut=pauses,
                words_final_voice=realigned, words_final_mix=in_mix, mix=mix_info,
                proofs=[f"{out_dir}/proof_shutter_alignment.png", f"{out_dir}/proof_timeline.png"])
