# Composes and renders original music to a film's structure: python3 music_make.py brief.json out.wav
#
# The music is cut to the film, not the film to the music: the tempo is chosen so bar lines land on the
# film's key moments, each section's energy follows the film's arc, and the last chord resolves on the end
# card. Royalty-free by construction. Brief:
#   {"bpm": 108, "key": "D", "mode": "major", "length": 37.0, "vibe": "calm-optimistic",
#    "progression": ["I", "V", "vi", "IV"],          # one chord per bar, repeated
#    "sections": [[0, 0], [2, 1], [4, 2], [7, 3], [14, 1], [15, 2], [16, "end"]]}   # [bar, energy 0-3 or "end"]
# Energy: 0 pad and piano, 1 + plucked arpeggio and hats, 2 + kick, bass and claps, 3 + fuller groove,
# 16th arpeggio, piano chords. A section starting at 2 or higher after a lower one gets a riser into it.
# "end" holds the tonic chord and lets it ring to the end. Vibes set the sounds and patterns:
#   calm-optimistic   warm piano, soft pad, harp-like pluck, light drums (human, nuchter, warm)
#   tech-minimal      pulsing pluck, glassy bells, tight kick, no piano (precise products, SaaS)
#   cinematic-warm    pads, piano, low strings, no drums, slower (emotional, premium)
#   playful           marimba-like pluck, claps, bouncy bass (consumer, fun)
# Needs numpy.
import json, sys, wave
import numpy as np

SR = 48000
rng = np.random.default_rng(3)
NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
ROMAN = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6}
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
VIBES = {
    "calm-optimistic": dict(piano=1.0, pad=0.8, pluck=0.7, bells=0.0, kick=0.8, clap=0.45, hats=0.35, bass=0.8, swing=0.04, bright=1.0),
    "tech-minimal":    dict(piano=0.0, pad=0.6, pluck=0.9, bells=0.5, kick=1.0, clap=0.35, hats=0.5, bass=0.9, swing=0.0, bright=1.2),
    "cinematic-warm":  dict(piano=1.0, pad=1.0, pluck=0.3, bells=0.2, kick=0.0, clap=0.0, hats=0.0, bass=0.6, swing=0.0, bright=0.8),
    "cinematic-pulse": dict(piano=0.55, pad=1.0, pluck=0.5, bells=0.22, kick=0.75, clap=0.0, hats=0.16, bass=0.85, swing=0.0, bright=0.8),   # MARSOPS: serious, modern, building
    "playful":         dict(piano=0.5, pad=0.5, pluck=1.0, bells=0.3, kick=0.9, clap=0.7, hats=0.5, bass=1.0, swing=0.12, bright=1.1),
}

def hz(m): return 440.0 * 2 ** ((m - 69) / 12)

def chord_notes(root_midi, mode, numeral):
    lower = numeral.lower() == numeral
    deg = ROMAN[numeral.upper()]
    sc = SCALES[mode]
    pcs = [sc[(deg + k) % 7] + 12 * ((deg + k) // 7) for k in (0, 2, 4)]
    return [root_midi + p for p in pcs], lower

def adsr(n, a, d, s, r, sr=SR):
    a, d, r = int(a * sr), int(d * sr), int(r * sr)
    e = np.full(n, s, dtype=np.float64)
    e[:a] = np.linspace(0, 1, max(a, 1))[: len(e[:a])]
    if a < n: e[a:a + d] = np.linspace(1, s, max(d, 1))[: len(e[a:a + d])]
    if r and n > r: e[-r:] *= np.linspace(1, 0, r)
    return e

# ------------------------------------------------------------------ instruments (mono note → array)
def piano(f, dur, vel=0.8):
    n = int((dur + 1.2) * SR); t = np.arange(n) / SR
    idx = 1.6 * np.exp(-t * 3.0) * vel                       # FM electric piano: index decays, tone softens
    y = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t)) + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 6)
    return y * np.exp(-t * (1.6 + f / 900)) * (1 - np.exp(-t * 400)) * vel

def pad(f, dur, bright=1.0):
    n = int((dur + 0.9) * SR); t = np.arange(n) / SR; y = np.zeros(n)
    for det in (-0.07, 0.0, 0.065):                          # three detuned voices, partials rolled off (a filtered saw)
        ff = f * 2 ** (det / 12)
        for k in range(1, 10):
            y += np.sin(2 * np.pi * ff * k * t + rng.uniform(0, 6.28)) / k * np.exp(-k / (3.2 * bright))
    return y * adsr(n, 0.45, 0.4, 0.75, 0.8) / 6

_pluck_cache = {}
def pluck(f, vel=0.7):
    """Karplus-Strong string: a burst of noise in a delay line that loses its highs (cached per pitch)."""
    key = round(f, 2)
    if key not in _pluck_cache: _pluck_cache[key] = _pluck(f)
    return _pluck_cache[key] * vel

def _pluck(f):
    n = int(1.2 * SR); p = max(2, int(SR / f)); buf = rng.uniform(-1, 1, p); y = np.zeros(n)
    for i in range(n):
        y[i] = buf[i % p]
        buf[i % p] = 0.5 * (buf[i % p] + buf[(i + 1) % p]) * 0.996
    return y * np.exp(-np.arange(n) / SR * 2.2)

def bells(f, vel=0.5):
    n = int(1.6 * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * t + 0.8 * np.sin(2 * np.pi * f * 3.5 * t) * np.exp(-t * 4)) * np.exp(-t * 2.5)) * vel

def bass(f, dur, vel=0.8):
    n = int((dur + 0.05) * SR); t = np.arange(n) / SR
    y = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t) + 0.1 * np.sin(2 * np.pi * 3 * f * t)
    return np.tanh(1.4 * y) * adsr(n, 0.008, 0.12, 0.7, 0.06) * vel

def kick(vel=1.0):
    n = int(0.45 * SR); t = np.arange(n) / SR; f = 48 + 90 * np.exp(-t / 0.035)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16) * vel

def noise_hp(n, cut):
    x = rng.standard_normal(n); X = np.fft.rfft(x); fr = np.fft.rfftfreq(n, 1 / SR)
    X *= 1 / (1 + (cut / np.maximum(fr, 1)) ** 4)
    return np.fft.irfft(X, n)

def clap(vel=0.6):
    n = int(0.3 * SR); t = np.arange(n) / SR
    env = sum(np.exp(-np.maximum(0, t - d) / 0.012) * (t >= d) for d in (0, 0.009, 0.018)) + 0.6 * np.exp(-t / 0.09)
    return noise_hp(n, 900) * env * vel * 0.5

def hat(vel=0.3, open_=False):
    n = int((0.25 if open_ else 0.06) * SR); t = np.arange(n) / SR
    return noise_hp(n, 7000) * np.exp(-t / (0.08 if open_ else 0.018)) * vel

def riser(length):
    n = int(length * SR); t = np.arange(n) / n
    x = noise_hp(n, 400) * t ** 2.5
    return x * 0.35

# ------------------------------------------------------------------ arrangement
def add(bus, y, start, gain=1.0, pan=0.0):
    s = int(start * SR)
    if s >= len(bus): return
    e = min(len(bus), s + len(y))
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    bus[s:e, 0] += y[: e - s] * gain * l * 1.41; bus[s:e, 1] += y[: e - s] * gain * r * 1.41

def reverb(x, length=2.2, wet=0.22):
    n = int(length * SR); ir = rng.standard_normal((n, 2)) * np.exp(-np.linspace(0, 6, n))[:, None]
    ir[: int(0.012 * SR)] = 0
    out = np.zeros((len(x) + n, 2))
    for c in range(2):
        L = len(x) + n; F = 1 << int(np.ceil(np.log2(L)))
        out[:, c] = np.fft.irfft(np.fft.rfft(x[:, c], F) * np.fft.rfft(ir[:, c], F), F)[:L]
    out *= wet / np.sqrt((ir ** 2).sum() / 2)
    out[: len(x)] += x
    return out[: len(x)]

def main():
    b = json.load(open(sys.argv[1])); out = sys.argv[2]
    bpm, length = b["bpm"], b["length"]; beat = 60 / bpm; bar = 4 * beat
    v = VIBES[b.get("vibe", "calm-optimistic")]
    root = 48 + NOTE[b["key"]]                                 # chords around C3–C4
    mode = b.get("mode", "major"); prog = b.get("progression", ["I", "V", "vi", "IV"])
    secs = b["sections"]; nbars = int(np.ceil(length / bar)) + 1
    energy = []
    for i in range(nbars):
        e = 0
        for s_bar, s_e in secs:
            if i >= s_bar: e = s_e
        energy.append(e)
    n = int((length + 3) * SR)
    music, drums = np.zeros((n, 2)), np.zeros((n, 2))
    side = np.ones(n)                                          # kick sidechain for pad and bass
    for i in range(nbars):
        t0 = i * bar; e = energy[i]
        if t0 >= length: break
        numeral = prog[i % len(prog)]
        notes, minor = chord_notes(root, mode, numeral)
        if e == "end":
            for k, m in enumerate(notes + [notes[0] + 12]):
                add(music, piano(hz(m + 12), 3.5, 0.7), t0 + k * 0.02, v["piano"] * 0.5, pan=-0.2 + 0.13 * k)
                add(music, pad(hz(m), length - t0 + 0.5, v["bright"]), t0, v["pad"] * 0.30)
            add(drums, kick(0.7), t0, v["kick"] * 0.7)
            continue
        # pad: whole chord, the whole bar
        for m in notes:
            add(music, pad(hz(m), bar, v["bright"]), t0, v["pad"] * (0.22 if e else 0.30), pan=rng.uniform(-0.4, 0.4))
        # piano: chord on 1 (and on 3 at energy 3); a melody note at energy 0
        if v["piano"]:
            hits = [0, 2] if e == 3 else [0]
            for h in hits:
                for k, m in enumerate(notes):
                    add(music, piano(hz(m + 12), beat * 1.5, 0.55 if h else 0.7), t0 + h * beat + k * 0.012, v["piano"] * 0.42, pan=-0.25 + 0.25 * k)
            if e <= 1:
                sc = SCALES[mode]; mel = [notes[2] + 12, notes[1] + 12, notes[0] + 24 if i % 2 else notes[2] + 12]
                for k, m in enumerate(mel):
                    add(music, piano(hz(m + 12), beat, 0.45), t0 + (1.5 + k) * beat, v["piano"] * 0.3, pan=0.3)
        # pluck arpeggio: 8ths at energy 1–2, 16ths at 3
        if e >= 1 and v["pluck"]:
            step = beat / (4 if e == 3 else 2); pattern = [0, 1, 2, 1, 2, 0, 1, 2]
            for k in range(int(bar / step)):
                m = notes[pattern[k % len(pattern)]] + 24 + (12 if (k % 8 == 6 and e == 3) else 0)
                sw = v["swing"] * step if k % 2 else 0
                add(music, pluck(hz(m), 0.5 + 0.2 * (k % 4 == 0)), t0 + k * step + sw, v["pluck"] * 0.33, pan=0.35 if k % 2 else -0.35)
        if e >= 1 and v["bells"]:
            add(music, bells(hz(notes[2] + 36), 0.4), t0 + 3 * beat, v["bells"] * 0.3, pan=0.5)
        # drums and bass
        if e >= 2:
            for q in range(4):
                if v["kick"] and (q in (0, 2) or e == 3):
                    add(drums, kick(0.95 if q == 0 else 0.8), t0 + q * beat, v["kick"] * 0.9)
                    s = int((t0 + q * beat) * SR); d = int(0.25 * SR)
                    side[s:s + d] = np.minimum(side[s:s + d], 1 - 0.45 * np.exp(-np.arange(min(d, n - s)) / (0.07 * SR)))
                if v["clap"] and q in (1, 3):
                    add(drums, clap(0.7), t0 + q * beat, v["clap"] * 0.8, pan=0.05)
            if v["bass"]:
                bm = notes[0] - 12 if notes[0] - 12 >= 36 else notes[0]
                pulses = [0, 1.5, 2, 3.5] if e == 2 else [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
                for p in pulses:
                    add(music, bass(hz(bm), beat * 0.45, 0.8 if p % 1 == 0 else 0.6), t0 + p * beat, v["bass"] * 0.55)
        if e >= 1 and v["hats"]:
            for k in range(8 if e < 3 else 16):
                stepk = bar / (8 if e < 3 else 16)
                add(drums, hat(0.25 + 0.15 * (k % 2 == 0), open_=(e == 3 and k % 4 == 2)), t0 + k * stepk + (v["swing"] * stepk if k % 2 else 0), v["hats"] * 0.6, pan=0.25)
        # a riser into a lift
        nxt = energy[i + 1] if i + 1 < nbars else 0
        if nxt != "end" and isinstance(nxt, int) and isinstance(e, int) and nxt >= 2 and nxt > e:
            add(music, riser(bar * 0.5), t0 + bar * 0.5, 0.5)
    music *= side[:, None] ** 0.8
    mix = reverb(music, 2.4, 0.26) + reverb(drums, 0.8, 0.08)
    mix = mix[: int(length * SR)]
    fade = int(min(2.5, length * 0.1) * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.3) * 0.85         # gentle saturation, then a ceiling
    pcm = (mix * 32767).astype("<i2")
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    print(f"{out}: {length:.1f} s, {bpm} BPM, {b['key']} {mode}, {b.get('vibe')}, bars {bar:.3f} s; energy by bar {energy[: int(np.ceil(length / bar))]}")

main()
