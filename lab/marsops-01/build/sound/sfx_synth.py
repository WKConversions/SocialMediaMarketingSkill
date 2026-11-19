# Makes the sound effects Karl's library doesn't have, so they are royalty-free and their sync point is
# exact: python3 sfx_synth.py <library/sound>   (adds sfx/<category>/gen-*.wav and their index entries)
#   whoosh   a band of noise that sweeps up into the moment of passing and away (camera moves, things
#            flying past, whip pans); sync = the moment of passing
#   riser    noise and a tone climbing into a reveal; sync = the end of the climb (the reveal)
#   impact   a soft low thump with a short click (a landing, a stamp, a logo locking in); sync = the hit
#   stamp    a thud with a paper slap; shimmer  bright partials sparkling (a reveal, a brand moment)
#   swell    soft air swelling and fading (a slow push, a scene breathing in)
# Stereo, 48 kHz, with a small synthetic room. Needs numpy.
import json, os, sys, wave
import numpy as np

SR = 48000
rng = np.random.default_rng(7)

def stft_filter(noise, fc, bw):
    """Noise through a band whose centre (Hz) and width (Hz) follow fc(t), bw(t), per 1024-sample frame."""
    N, H = 2048, 512
    win = np.hanning(N)
    out = np.zeros(len(noise) + N)
    freqs = np.fft.rfftfreq(N, 1 / SR)
    for k, s in enumerate(range(0, len(noise) - N, H)):
        t = (s + N / 2) / len(noise)
        spec = np.fft.rfft(noise[s:s + N] * win)
        c, b = fc(t), bw(t)
        mask = np.exp(-0.5 * ((np.log2(np.maximum(freqs, 20)) - np.log2(c)) / max(b, 1e-3)) ** 2)
        out[s:s + N] += np.fft.irfft(spec * mask) * win
    return out[: len(noise)] / 1.5

def room(x, length=0.8, wet=0.18):
    n = int(length * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir[0] = 0
    y = np.convolve(x, ir)[: len(x) + n] * (wet / np.sqrt(np.sum(ir ** 2)))
    y[: len(x)] += x
    return y

def stereo(f, *a, **k):
    L, R = f(*a, **k), f(*a, **k)       # two independent noise draws: natural width
    m = max(len(L), len(R)); L = np.pad(L, (0, m - len(L))); R = np.pad(R, (0, m - len(R)))
    return np.stack([L, R], 1)

def whoosh(length=0.6, sync=0.38, lo=300, hi=3500):
    n = int(length * SR); t = np.arange(n) / n; ts = sync / length
    noise = rng.standard_normal(n)
    fc = lambda u: lo * (hi / lo) ** (np.exp(-((u - ts) / 0.22) ** 2))
    y = stft_filter(noise, fc, lambda u: 0.9 - 0.4 * np.exp(-((u - ts) / 0.2) ** 2))
    env = np.where(t < ts, (t / ts) ** 2.6, np.exp(-(t - ts) / (1 - ts) * 4.5))
    return room(y * env, 0.5, 0.12)

def riser(length=1.5, lo=180, hi=1400):
    n = int(length * SR); t = np.arange(n) / n
    noise = stft_filter(rng.standard_normal(n), lambda u: 400 * (8 ** u), lambda u: 0.7)
    f = lo * (hi / lo) ** (t ** 1.6)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.6 + 0.4 * np.sin(2 * np.pi * (4 + 10 * t) * t * length))
    y = (0.55 * noise + 0.35 * tone) * t ** 2.2
    y[-int(0.03 * SR):] *= np.linspace(1, 0, int(0.03 * SR))
    return room(y, 0.9, 0.2)

def impact(length=1.2, f0=110, f1=42, click=0.5):
    n = int(length * SR); tt = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-tt / 0.045)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.28)
    cl = stft_filter(rng.standard_normal(n), lambda u: 2500, lambda u: 1.2) * np.exp(-tt / 0.012) * click
    return room(0.9 * body + cl, 1.0, 0.15)

def stamp(length=0.7):
    n = int(length * SR); tt = np.arange(n) / SR
    f = 75 + 90 * np.exp(-tt / 0.03)
    thud = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.09)
    slap = stft_filter(rng.standard_normal(n), lambda u: 1800, lambda u: 1.0) * np.exp(-tt / 0.02)
    return room(0.8 * thud + 0.7 * slap, 0.4, 0.1)

def shimmer(length=1.4):
    n = int(length * SR); tt = np.arange(n) / SR; y = np.zeros(n)
    for k in range(9):
        f = 2093 * 2 ** (rng.choice([0, 4, 7, 12, 16, 19, 24]) / 12)
        t0 = k * 0.035; e = np.where(tt > t0, np.exp(-(tt - t0) / 0.35) * (1 - np.exp(-(tt - t0) / 0.004)), 0)
        y += np.sin(2 * np.pi * f * tt) * e * 0.25
    return room(y, 1.4, 0.35)

def swell(length=1.6, peak=0.9):
    n = int(length * SR); t = np.arange(n) / n; tp = peak / length
    y = stft_filter(rng.standard_normal(n), lambda u: 700 + 900 * np.exp(-((u - tp) / 0.3) ** 2), lambda u: 1.4)
    env = np.where(t < tp, np.sin(np.pi / 2 * t / tp) ** 2, np.cos(np.pi / 2 * (t - tp) / (1 - tp)) ** 2)
    return room(y * env, 1.0, 0.2)

SPECS = [  # id, category, character, maker, kwargs, sync (s)
    ("gen-whoosh-short", "whoosh", "short, soft pass: a card or a word flying in", whoosh, dict(length=0.45, sync=0.26), 0.26),
    ("gen-whoosh-medium", "whoosh", "medium pass: a camera move, a page pan", whoosh, dict(length=0.8, sync=0.5, lo=220, hi=2800), 0.5),
    ("gen-whoosh-whip", "whoosh", "fast, bright: a whip pan", whoosh, dict(length=0.35, sync=0.17, lo=500, hi=6000), 0.17),
    ("gen-whoosh-deep", "whoosh", "slow, low: a big pull-back or zoom", whoosh, dict(length=1.2, sync=0.8, lo=120, hi=1500), 0.8),
    ("gen-riser-1s", "riser", "one second into a reveal", riser, dict(length=1.0), 1.0),
    ("gen-riser-2s", "riser", "two seconds of build into a reveal", riser, dict(length=2.0), 2.0),
    ("gen-impact-soft", "impact", "soft low thump: a landing, a lock-in", impact, dict(click=0.35), 0.0),
    ("gen-impact-deep", "impact", "deeper, longer: a logo, the end card", impact, dict(length=1.8, f0=90, f1=34, click=0.5), 0.0),
    ("gen-stamp", "impact", "thud with a paper slap: a stamp", stamp, dict(), 0.0),
    ("gen-shimmer", "stinger", "bright sparkle: a reveal, a brand moment", shimmer, dict(), 0.0),
    ("gen-swell", "swipe", "soft air swelling: a slow push, a scene breathing in", swell, dict(), 0.9),
]

def write(path, y):
    y = y / (np.abs(y).max() + 1e-9) * 0.7
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())

def main():
    lib = sys.argv[1]
    idx_path = os.path.join(lib, "sfx", "index.json")
    idx = [o for o in json.load(open(idx_path)) if not o["id"].startswith("gen-")]
    for sid, cat, character, maker, kw, sync in SPECS:
        y = stereo(maker, **kw)
        rel = os.path.join("sfx", cat, sid + ".wav"); os.makedirs(os.path.join(lib, "sfx", cat), exist_ok=True)
        write(os.path.join(lib, rel), y)
        idx.append({"id": sid, "file": rel, "category": cat, "character": character, "licence": "generated", "sync": sync,
                    "transients": [sync], "length": round(len(y) / SR, 3), "lufs": None, "hits": 1, "original": ["scripts/sfx_synth.py"]})
    json.dump(idx, open(idx_path, "w"), indent=1)
    cat_path = os.path.join(lib, "catalogue.md")
    text = open(cat_path).read().split("\n## generated")[0].rstrip() + "\n\n## generated\n\nMade by `scripts/sfx_synth.py` for what the folder lacks; royalty-free, sync exact.\n\n"
    text += "| id | category | character | sync s | length s |\n|---|---|---|---|---|\n"
    text += "\n".join(f"| `{o['id']}` | {o['category']} | {o['character']} | {o['sync']:.2f} | {o['length']:.2f} |" for o in idx if o["id"].startswith("gen-")) + "\n"
    open(cat_path, "w").write(text)
    print(f"{len(SPECS)} generated sounds added to {lib}/sfx")

if __name__ == "__main__":
    main()
