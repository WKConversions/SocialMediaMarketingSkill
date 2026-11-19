"""Sound for the MARSOPS teaser: the composed bed (music.wav) + effects placed on the picture's own cue schedule
(cues.json, exported from render.html): Geiger clicks on every strike and the background count until the suit's
shield closes, a soft 'tink' when a ray stops on a shield, a shimmer as the suit's shield closes (7.15 s), a riser
into the impact where the line becomes MARSOPS (12.0 s). Mastered to -14 LUFS, true peak under -1 dBTP.
Usage: python3 mix_teaser.py cues.json music.wav out.wav
"""
import json
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, '.')
import sfx_synth as sx  # noqa: E402

SR = 48000
LEN = 15.0
rng = np.random.default_rng(1882)


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def place(bus, y, t, gain_db=0.0, pan=0.0):
    if y.ndim == 1:
        y = np.stack([y * np.sqrt(0.5 * (1 - pan)), y * np.sqrt(0.5 * (1 + pan))], axis=1) * np.sqrt(2)
    i = int(round(t * SR))
    if i < 0:
        y, i = y[-i:], 0
    j = min(len(bus), i + len(y))
    bus[i:j] += y[: j - i] * 10 ** (gain_db / 20)


def geiger_click(amp=1.0):
    """A Geiger-counter tick: a 1 ms broadband impulse with a short resonant ring of the tube's case."""
    n = int(0.012 * SR)
    t = np.arange(n) / SR
    imp = rng.normal(0, 1, n) * np.exp(-t / 0.0007)
    ring = np.sin(2 * np.pi * (2900 + rng.uniform(-200, 200)) * t) * np.exp(-t / 0.0035) * 0.5
    y = imp + ring
    return y / np.abs(y).max() * amp


def tink(amp=1.0):
    """A ray stopping on the shield: two soft high partials, quick decay (glass-like, quiet)."""
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * 2350 * t) + 0.6 * np.sin(2 * np.pi * 3520 * t + 1.0)) * np.exp(-t / 0.09) * (1 - np.exp(-t / 0.002))
    return y / np.abs(y).max() * amp


def sync_peak(y):
    m = np.abs(y).mean(axis=1) if y.ndim == 2 else np.abs(y)
    return int(np.argmax(m))


def main(cues_path, music_path, out):
    c = json.load(open(cues_path))
    n = int(LEN * SR)
    music = load(music_path)[:n]
    music = np.pad(music, ((0, n - len(music)), (0, 0)))
    fx = np.zeros((n + SR, 2))
    strikes = {round(s['t'], 3) for s in c['strikes'] if s['kind'] == 'ink'}
    for t in c['clicks']:                                          # the count: strikes a little louder than the background
        g = -20 if round(t, 3) in strikes else -27 + rng.uniform(-3, 2)
        place(fx, geiger_click(), t, g, rng.uniform(-0.5, 0.5))
    for s in c['strikes']:
        if s['kind'] == 'neon':
            place(fx, tink(), s['t'], -31 + rng.uniform(-2, 1), rng.uniform(-0.3, 0.3))
    sh = sx.shimmer(1.4)
    place(fx, sh, c['shieldClose'] - 0.05, -22)
    r = sx.riser(1.6)
    place(fx, r, c['logo'] - len(r) / SR, -20)                    # the riser ends on the logo
    im = sx.impact(1.6)
    place(fx, im, c['logo'] - sync_peak(im) / SR, -11)
    mix = music + fx[:n]
    # master: measure, set to -14 LUFS, then a true-peak-safe limiter (latency compensated)
    tmp = out.replace('.wav', '_pre.wav')
    write(tmp, mix / (np.abs(mix).max() + 1e-9) * 0.5)
    lufs = loudness(tmp)
    gain = -14.0 - lufs
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", f"volume={gain:.2f}dB,alimiter=limit=0.84:level=0:latency=1,aresample=48000",
                    "-c:a", "pcm_s16le", out], check=True)
    print(f"premaster {lufs:.1f} LUFS -> gain {gain:+.1f} dB; final {loudness(out):.1f} LUFS -> {out}")


def write(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def loudness(path):
    err = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    import re
    s = err[err.rfind('Summary:'):]
    return float(re.search(r"I:\s+(-?[\d.]+) LUFS", s).group(1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
