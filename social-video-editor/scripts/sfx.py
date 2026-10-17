"""The caption effects track, placed from the plan so every sound sits on its caption's first frame.

Rules (client, PREFERENCES.md): a sound only where a new caption appears and on EVERY caption; the caption
sound is the client's own camera shutter with all its clicks, and the shutter's biggest wave lands exactly
on the first frame the caption is visible (ceil(start * fps) / fps, not the planned word time); the softer
clicks come just before. Script captions add a short synthesised hit that starts on that frame. Pitch,
level and pan of each use come from the plan. No magical/glowy/chime sounds.

The biggest wave is one fixed point of the file: its loudest sample (~194.5 ms in). Each use is repitched
on a sample grid that passes through that sample, so it keeps its exact value, stays the loudest sample of
the sound as played (linear interpolation never overshoots it) and lands exactly on the caption's frame
sample, whatever the pitch. (A grid from sample 0 can miss the spike, and the loudest played sample then
jumps to a neighbouring half-cycle up to ~2 ms away.)
"""
import wave

import numpy as np

from common import decode_audio, first_frame, skill_path


def db(x):
    return 10 ** (x / 20)


# ---------------------------------------------------------------------------------------- shutter
def load_shutter(cfg, sr):
    """The client's shutter: the slice with the clicks, short fades, peak-normalised (mono).
    Returns (sound, anchor): anchor = index of the biggest wave, the file's loudest sample."""
    x = decode_audio(skill_path(cfg["file"]), sr).astype(float)
    a, b = (int(t * sr) for t in cfg["slice_s"])
    x = x[a:b].copy()
    fi, fo = int(cfg["fade_in_s"] * sr), int(cfg["fade_out_s"] * sr)
    x[:fi] *= np.linspace(0, 1, fi)
    x[-fo:] *= np.linspace(1, 0, fo)
    return x / np.abs(x).max(), int(np.argmax(np.abs(x)))


def repitch(x, factor, anchor):
    """Resample by factor (> 1 = higher and shorter) on a grid through sample `anchor`, which keeps its
    exact value. Returns (sound, index of the anchor in it)."""
    before = int(anchor // factor)
    pos = anchor + factor * np.arange(-before, int((len(x) - 1 - anchor) // factor) + 1)
    return np.interp(pos, np.arange(len(x)), x), before


# ---------------------------------------------------------------------------------------- hit
def bandpass_noise(rng, n, lo, hi, sr):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    spec *= ((f >= lo) & (f <= hi)).astype(float) * np.exp(-((np.log(np.maximum(f, 1) / np.sqrt(lo * hi))) ** 2) / 2)
    out = np.fft.irfft(spec, n)
    return out / (np.abs(out).max() + 1e-9)


def hit(size, cfg, rng, sr):
    """Sub-bass drop + mid thud + knock + click (phone speakers barely play below ~150 Hz)."""
    dur = cfg["duration_s"] * size
    n = int(dur * sr)
    t = np.arange(n) / sr
    s, th, kn, cl = cfg["sub"], cfg["thud"], cfg["knock"], cfg["click"]
    freq = s["f_end_hz"] + (s["f_start_hz"] - s["f_end_hz"]) * np.exp(-t / s["glide_s"])
    sub = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t / (s["decay_s"] * size))
    sub = np.tanh(s["drive"] * sub) / np.tanh(s["drive"])
    thud_f = th["f_hz"] + th["f_glide_hz"] * np.exp(-t / th["glide_s"])
    thud = np.sin(2 * np.pi * np.cumsum(thud_f) / sr) * np.exp(-t / th["decay_s"])
    knock = bandpass_noise(rng, n, *kn["band_hz"], sr) * np.exp(-t / kn["decay_s"])
    click = bandpass_noise(rng, n, *cl["band_hz"], sr) * np.exp(-t / cl["decay_s"])
    x = s["level"] * sub + th["level"] * thud + kn["level"] * knock + cl["level"] * click
    x *= np.exp(-t / (cfg["body_decay_s"] * size)) ** 0.5 * np.clip((dur - t) / cfg["fade_out_s"], 0, 1)
    return x / np.abs(x).max()


def reverb(x, cfg, rng, sr):
    """Stereo noise-tail reverb -> (n, 2)."""
    n = int(cfg["length_s"] * sr)
    t = np.arange(n) / sr
    out = []
    for _ in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / cfg["decay_s"])
        ir = ir / np.sqrt((ir ** 2).sum())
        m = len(x) + n
        y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)
        out.append((1 - cfg["wet"]) * np.pad(x, (0, n)) + cfg["wet"] * y * cfg["wet_gain"])
    return np.stack(out, axis=1)


# ---------------------------------------------------------------------------------------- placement
def pan_gains(pan):
    """Equal-power pan of a mono sound: (left, right) gains, both 1 in the centre."""
    return np.sqrt(1 - pan), np.sqrt(1 + pan)


def place(track, sound, i0, gain_db, pan=0.0):
    """Add a mono (equal-power panned) or stereo sound so that its sample 0 lands on track sample i0."""
    if sound.ndim == 1:
        sound = np.stack([sound * g for g in pan_gains(pan)], axis=1)
    if i0 < 0:
        sound, i0 = sound[-i0:], 0
    end = min(len(track), i0 + len(sound))
    if end > i0:
        track[i0:end] += sound[: end - i0] * db(gain_db)


def onset(sound, rel_db):
    """Index of the first sample within rel_db of the sound's peak."""
    mono = np.abs(sound if sound.ndim == 1 else sound.mean(axis=1))
    return int(np.argmax(mono > mono.max() * db(rel_db)))


def shutter_click_lead_s(cfg, pitch, sr, rel_db):
    """How long (s) before its biggest wave the shutter's first click comes (its first sample within rel_db of
    the biggest wave), played at pitch. With the biggest wave on a caption's first frame, that caption must
    appear at least this long after the start of the video or the clicks before it are cut off."""
    base, anchor = load_shutter(cfg, sr)
    snd, peak = repitch(base, pitch, anchor)
    return (peak - onset(snd, rel_db)) / sr


def write_wav(path, x, sr):
    """16-bit stereo WAV."""
    x = np.clip(x, -1, 1)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def frame_sample(frame, fps, sr):
    """The audio sample at which a video frame starts."""
    return int(round(frame * sr / fps))


def build(plan, style, out_track):
    """Write the effects track; return what was placed where."""
    cfg = style["sfx"]
    sr, fps = cfg["sample_rate"], plan["clip"]["fps"]
    n = frame_sample(plan["clip"]["frames"], fps, sr)
    track = np.zeros((n + sr, 2))
    shutter, anchor = load_shutter(cfg["shutter"], sr)
    placed = []
    for k, c in enumerate(plan["captions"]):
        f = first_frame(c["start"], fps)                       # the caption becomes visible on this frame
        i = frame_sample(f, fps, sr)
        sh = c["sfx"]["shutter"]
        snd, peak = repitch(shutter, sh["pitch"], anchor)
        place(track, snd, i - peak, sh["gain_db"], sh["pan"])  # the biggest wave on the caption frame
        entry = dict(caption=k, text=c["text"], frame=f, frame_sample=i, shutter_start_sample=i - peak,
                     shutter_peak_sample=i - peak + int(np.argmax(np.abs(snd))))
        h = c["sfx"]["hit"]
        if h:
            hc = cfg["hit"]
            rng = np.random.default_rng([cfg["seed"], k])
            snd_h = reverb(hit(h["size"] * hc["synth"]["size_scale"], hc["synth"], rng, sr),
                           hc["reverb"]["tail" if h["tail"] else "short"], rng, sr)
            place(track, snd_h, i - onset(snd_h, hc["onset_rel_db"]), h["gain_db"])
            entry["hit"] = h["role"]
        placed.append(entry)
    track = track[:n]
    fade = int(cfg["track_fade_out_s"] * sr)
    track[-fade:] *= np.linspace(1, 0, fade)[:, None]
    write_wav(out_track, track, sr)
    return dict(sample_rate=sr, peak_dbfs=round(float(20 * np.log10(np.abs(track).max() + 1e-12)), 2), placed=placed)
