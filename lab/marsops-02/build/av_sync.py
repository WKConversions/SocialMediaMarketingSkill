# Measures how sound hits sit against picture hits: for every strong visual hit (a cut, a landing, a flash), the
# nearest audio onset within ±8 frames, and its offset in frames (negative = the sound comes first).
#   python3 av_sync.py film.mp4 [audio.wav]      (audio from the film itself when no wav is given)
#   [--hits 20] [--json out.json]
# Visual hits: peaks of the frame-to-frame change (top --hits, at least 6 frames apart). Audio onsets: positive
# spectral flux, per video frame. Reports each hit's offset and the median, and how many hits have a sound at all.
# Needs numpy, opencv; ffmpeg to pull audio from a film.
import sys, os, json, subprocess, tempfile, wave, cv2, numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]
def opt(n, d): return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
nh = int(opt("--hits", 20)); film = args[0]
if "--hits" in sys.argv: args = [a for a in args if a != opt("--hits", "")]
if "--json" in sys.argv: args = [a for a in args if a != opt("--json", "")]
wav = args[1] if len(args) > 1 else None
if not wav:
    wav = os.path.join(tempfile.mkdtemp(), "a.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", film, "-vn", "-ac", "1", "-ar", "22050", wav], check=True)
cap = cv2.VideoCapture(film); fps = cap.get(cv2.CAP_PROP_FPS) or 30; prev = None; d = []
while True:
    ok, im = cap.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(im, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    d.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g
d = np.array(d)
w = wave.open(wav); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
hop = sr / fps; win = 1024; n = len(d); flux = np.zeros(n); last = None
for i in range(n):
    c = int(i * hop); seg = x[c:c + win]
    if len(seg) < win: seg = np.pad(seg, (0, win - len(seg)))
    mag = np.abs(np.fft.rfft(seg * np.hanning(win)))
    if last is not None: flux[i] = np.maximum(mag - last, 0).sum()
    last = mag
flux = flux / (np.percentile(flux, 95) + 1e-9)
# visual hits: local maxima of d, strongest first, 6 frames apart, ignoring a fade-in at the very start
order = np.argsort(-d); hits = []
for i in order:
    if i < 2 or d[i] < 2 * np.median(d) + 1: continue
    if all(abs(i - h) >= 6 for h in hits): hits.append(int(i))
    if len(hits) >= nh: break
hits.sort(); rows = []
for h in hits:
    lo, hi = max(0, h - 8), min(n, h + 9); seg = flux[lo:hi]; k = int(np.argmax(seg))
    rows.append({"t": round(h / fps, 3), "frame": h, "visual": round(float(d[h]), 1),
                 "sound": round(float(seg[k]), 2), "offset": (lo + k) - h if seg[k] > 0.5 else None})
with_sound = [r for r in rows if r["offset"] is not None]
offs = np.array([r["offset"] for r in with_sound])
for r in rows: print(f"{r['t']:7.3f}s f{r['frame']:<5} visual {r['visual']:6.1f}  " + (f"sound {r['sound']:.2f} at {r['offset']:+d} frames" if r["offset"] is not None else "no sound hit"))
if len(offs):
    print(f"{os.path.basename(os.path.dirname(film)) or film}: {len(with_sound)}/{len(rows)} visual hits have a sound hit; offset median "
          f"{np.median(offs):+.0f} frames (negative = sound first), on the frame ±1: {100 * (np.abs(offs) <= 1).mean():.0f}%, "
          f"sound 2+ frames early: {100 * (offs <= -2).mean():.0f}%, late: {100 * (offs >= 2).mean():.0f}%")
# chance baseline: the same search at random frames. With music there is an onset near almost any frame, so a
# hit only counts as synced by how much better it does than chance.
rng = np.random.default_rng(0); rnd = []
for h in rng.integers(8, n - 9, 400):
    seg = flux[h - 8:h + 9]; k = int(np.argmax(seg)); rnd.append(k - 8 if seg[k] > 0.5 else 99)
rnd = np.array(rnd); real = np.array([r["offset"] if r["offset"] is not None else 99 for r in rows])
pr, pc = (np.abs(real) <= 1).mean(), (np.abs(rnd) <= 1).mean()
strength = [r["sound"] for r in rows]
peak_here = np.mean([flux[max(0, h - 1):h + 2].max() for h in hits]); peak_rand = np.mean([flux[h - 1:h + 2].max() for h in rng.integers(8, n - 9, 400)])
print(f"against chance: within ±1 frame {100 * pr:.0f}% of hits vs {100 * pc:.0f}% of random frames; sound energy at the hit "
      f"{peak_here:.2f} vs {peak_rand:.2f} at random frames (×{peak_here / max(peak_rand, 1e-9):.2f}) → "
      + ("hits are scored" if pr > pc + 0.15 or peak_here > 1.3 * peak_rand else "hits are NOT scored beyond what the music gives anyway"))
# beat grid: films cut to music land cuts on the beat even where no drum hits. Tempo from the autocorrelation of the
# onset curve (70–180 bpm), phase from the strongest alignment, then each hit's distance to the nearest beat.
fl = flux - flux.mean(); ac = np.correlate(fl, fl, "full")[len(fl) - 1:]
lo_l, hi_l = int(fps * 60 / 180), int(fps * 60 / 70); lag = lo_l + int(np.argmax(ac[lo_l:hi_l + 1])); per = float(lag)
# refine the period to a fraction of a frame
best = (-1, per, 0)
for pp in np.linspace(per - 1, per + 1, 41):
    for ph in np.linspace(0, pp, int(pp * 4), endpoint=False):
        idxs = np.arange(ph, n, pp).astype(int); sc = flux[idxs].mean()
        if sc > best[0]: best = (sc, pp, ph)
_, per, ph = best
def dist(h): x = (h - ph) % per; return min(x, per - x)
dh = np.array([dist(h) for h in hits]); dr = np.array([dist(h) for h in rng.integers(8, n - 9, 400)])
print(f"beat grid: {60 * fps / per:.1f} bpm (a beat every {per:.1f} frames); hits within 1 frame of a beat {100 * (dh <= 1).mean():.0f}% "
      f"vs {100 * (dr <= 1).mean():.0f}% of random frames; median distance {np.median(dh):.1f} vs {np.median(dr):.1f} frames → "
      + ("cut to the beat" if (dh <= 1).mean() > (dr <= 1).mean() + 0.15 else "not cut to the beat"))
if "--json" in sys.argv: json.dump(rows, open(opt("--json", "av.json"), "w"), indent=1)
