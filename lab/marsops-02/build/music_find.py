# Finds royalty-free music for a film and measures it, so the choice is made on facts, not titles.
#   python3 music_find.py --mood positive,hopeful,confident --feel bright,uplifting --bpm 95-120 --min 40 --out music/
#        [--n 8] [--avoid children,country]   (matched against genre and title; seasonal titles are avoided by default)
# Mixkit moods include positive, hopeful, confident, motivating, calm, relaxed, cheerful, friendly,
# dreamy, elegant, energetic, inspiring-like 'motivating', futuristic, atmospheric (see mixkit.co/free-stock-music).
#
# Sources that work from the build sandbox (others block scripts behind a browser check):
#   Mixkit       mixkit.co mood and tag pages carry each track's name, genre, artist, length and file;
#                Mixkit Stock Music Free License: commercial use, no credit needed (best for client work)
#   Incompetech  Kevin MacLeod's catalogue (pieces.json) carries tempo, feel, genre and instruments;
#                CC BY 4.0: credit "Music by Kevin MacLeod (incompetech.com), CC BY 4.0" in the description
# For each candidate it downloads the file and measures tempo (BPM), loudness, brightness, how busy it is
# (onsets per second), how much it builds (energy at the end against the start) and where the intro
# ends; it writes candidates.json, the files, and a sheet of every track's energy curve over time.
# Pick by the brief: tempo near the film's cut rhythm, a feel that matches the tone, busy-ness under the
# voice (a dense track fights the words), and a section whose shape fits the film's arc.
import json, os, re, subprocess, sys, urllib.request
import numpy as np

SR = 22050
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

def get(url, path=None):
    req = urllib.request.Request(url, headers=UA)
    data = urllib.request.urlopen(req, timeout=60).read()
    if path: open(path, "wb").write(data)
    return data

def iso_seconds(d):
    m = re.match(r"PT(?:(\d+)M)?(?:(\d+)S)?", d or "")
    return (int(m.group(1) or 0) * 60 + int(m.group(2) or 0)) if m else 0

def mixkit(moods):
    out = []
    for mood in moods:
        for kind in ("mood", "tag", "genre"):
            try:
                html = get(f"https://mixkit.co/free-stock-music/{kind}/{mood}/").decode("utf8", "ignore")
            except Exception:
                continue
            for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S):
                try: data = json.loads(block)
                except Exception: continue
                for node in data.get("@graph", []):
                    for it in node.get("itemListElement", []) if node.get("@type") == "ItemList" else []:
                        if it.get("@type") == "MusicRecording":
                            out.append({"source": "mixkit", "title": it["name"], "artist": it.get("byArtist"), "genre": it.get("genre"), "url": it["url"],
                                        "length": iso_seconds(it.get("duration")), "licence": "Mixkit Stock Music Free License (no credit needed)", "query": f"{kind}:{mood}"})
            break
    return out

def incompetech(feels, bpm):
    cat = json.loads(get("https://incompetech.com/music/royalty-free/pieces.json"))
    out = []
    for p in cat:
        f = [x.strip().lower() for x in (p.get("feel") or "").split(",")]
        try: b = int(p.get("bpm") or 0)
        except ValueError: b = 0
        hit = len(set(f) & set(feels))
        if hit and (not bpm or bpm[0] <= b <= bpm[1]):
            h, m, s = (p.get("length") or "0:0:0").split(":")
            out.append({"source": "incompetech", "title": p["title"], "artist": "Kevin MacLeod", "genre": p.get("instruments"), "feel": p.get("feel"),
                        "bpm_listed": b, "url": "https://incompetech.com/music/royalty-free/mp3-royaltyfree/" + urllib.request.quote(p["filename"]),
                        "length": int(h) * 3600 + int(m) * 60 + int(s), "licence": "CC BY 4.0: credit Kevin MacLeod (incompetech.com)", "match": hit,
                        "description": p.get("description")})
    return sorted(out, key=lambda x: -x["match"])

def analyse(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    hop = 512; n = len(x) // hop
    fr = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    spec = np.abs(np.fft.rfft(fr * np.hanning(hop), axis=1))
    flux = np.maximum(0, np.diff(spec, axis=0)).sum(axis=1); flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    # tempo: autocorrelation of the onset curve over 60–180 BPM, folded into 80–160
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    fps = SR / hop
    lags = np.arange(int(fps * 60 / 180), int(fps * 60 / 60))
    bpm = 60 * fps / lags[np.argmax(ac[lags])]
    while bpm < 80: bpm *= 2
    while bpm > 160: bpm /= 2
    onsets = int(np.sum((flux[1:-1] > 1.5) & (flux[1:-1] > flux[:-2]) & (flux[1:-1] > flux[2:])))
    freqs = np.fft.rfftfreq(hop, 1 / SR)
    bright = float(np.median((spec * freqs).sum(axis=1) / (spec.sum(axis=1) + 1e-9)))
    sec = int(fps); per_s = rms[: len(rms) // sec * sec].reshape(-1, sec).mean(axis=1)
    db = 20 * np.log10(per_s / per_s.max() + 1e-9)
    intro = int(np.argmax(db > -6)) if np.any(db > -6) else 0
    third = max(1, len(per_s) // 3)
    return {"bpm": round(float(bpm), 1), "onsets_per_s": round(onsets / (len(x) / SR), 2), "brightness_hz": round(bright),
            "build_db": round(float(20 * np.log10(per_s[-third:].mean() / per_s[:third].mean())), 1), "intro_s": intro, "energy_db_per_s": [round(float(v), 1) for v in db]}

def sheet(cands, path):
    from PIL import Image, ImageDraw
    W, H, L = 1400, 90, 420
    im = Image.new("RGB", (W + L, H * len(cands)), "white"); d = ImageDraw.Draw(im)
    for r, c in enumerate(cands):
        e = np.array(c["analysis"]["energy_db_per_s"]); y0 = r * H
        d.rectangle([L, y0 + 2, L + W, y0 + H - 2], fill=(246, 247, 250))
        for i, v in enumerate(e[: W // 6]):
            h = (v + 30) / 30 * (H - 10)
            d.rectangle([L + i * 6, y0 + H - 5 - max(0, h), L + i * 6 + 4, y0 + H - 5], fill=(60, 140, 120))
        a = c["analysis"]
        d.text((6, y0 + 6), f"{c['title']} ({c['source']})"[:60], fill=(0, 0, 0))
        d.text((6, y0 + 24), f"{a['bpm']} BPM  {a['onsets_per_s']} onsets/s  {a['brightness_hz']} Hz  build {a['build_db']:+} dB", fill=(60, 60, 60))
        d.text((6, y0 + 42), f"{c.get('genre') or ''}"[:60], fill=(110, 110, 110))
        d.text((6, y0 + 60), f"{c['length']} s  intro {a['intro_s']} s  score {c['score']:.2f}", fill=(110, 110, 110))
    im.save(path)

def main():
    out = arg("--out", "music"); os.makedirs(out, exist_ok=True)
    moods = [m for m in arg("--mood", "happy").split(",") if m]
    feels = [f.lower() for f in arg("--feel", "bright,uplifting").split(",") if f]
    b = arg("--bpm"); bpm = tuple(int(v) for v in b.split("-")) if b else None
    minlen, n = int(arg("--min", 30)), int(arg("--n", 8))
    avoid = [a.lower() for a in arg("--avoid", "children,christmas,xmas,holiday,halloween").split(",") if a]   # genre or title words
    ok = lambda c: c["length"] >= minlen and not any(a in (c.get("genre") or "").lower() + " " + c["title"].lower() for a in avoid)
    cands = [c for c in mixkit(moods) if ok(c)][: n] + [c for c in incompetech(feels, bpm) if ok(c)][: n // 2]
    seen, keep = set(), []
    for c in cands:
        if c["url"] in seen: continue
        seen.add(c["url"])
        fn = os.path.join(out, re.sub(r"[^a-z0-9]+", "-", c["title"].lower()).strip("-") + ".mp3")
        try:
            if not os.path.exists(fn): get(c["url"], fn)
            c["file"] = fn; c["analysis"] = analyse(fn)
        except Exception as e:
            print("skip", c["title"], e); continue
        a = c["analysis"]
        mid = (bpm[0] + bpm[1]) / 2 if bpm else a["bpm"]
        # closeness of tempo, a calm density under a voice, and a track that doesn't fall away
        c["score"] = round(max(0, 1 - abs(a["bpm"] - mid) / 40) + max(0, 1 - a["onsets_per_s"] / 6) * 0.5 + (0.3 if a["build_db"] > -3 else 0), 2)
        keep.append(c)
    keep.sort(key=lambda c: -c["score"])
    json.dump(keep, open(os.path.join(out, "candidates.json"), "w"), indent=1)
    sheet(keep, os.path.join(out, "candidates.png"))
    for c in keep:
        a = c["analysis"]
        print(f"{c['score']:.2f}  {c['title'][:34]:34s} {c['source']:11s} {a['bpm']:6.1f} BPM {a['onsets_per_s']:4.1f}/s {a['brightness_hz']:5d} Hz build {a['build_db']:+5.1f}  {c['length']} s  {c.get('genre') or ''}")

main()
