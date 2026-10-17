"""Creative decisions for v1 of the Brooklyn Bridge -> WKC edit, written to out/edit_v14.json + out/v14_events.json.

Times are OUTPUT seconds (after the uniform-pacing cuts in out/edl.json).
"""
import json
import os
import subprocess

E = "/home/user/SocialMediaMarketingSkill/lab/edit-04"
FPS = 30
NARA = f"{E}/broll/elephants/downloads/ia_fc-fc-50/fc-fc-50.mp4"   # National Archives NYC film (public domain)

# archival inserts cut from the NARA film: 4:3 frame scaled to the full width, centred on black
CUTS = {"bridge_wide": 150.5, "bridge_under": 135.0, "street": 125.0, "horse_cart": 154.0}


def make_insert(name, start, dur=6.0):
    out = f"{E}/broll/v14_{name}.mp4"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(dur), "-i", NARA, "-vf",
                        "fps=30,scale=1080:720:flags=lanczos,pad=1080:1920:0:600:black,format=yuv420p", "-an",
                        "-c:v", "libx264", "-crf", "14", out], check=True)
    return out


CAPTIONS = [  # (text, first word start, class, fill)
    ("Nobody", 0.08, "script", "WHITE"), ("trusted", 0.54, "sans", ""), ("the opening", 1.10, "sans", ""),
    ("of the", 1.74, "sans", ""), ("Brooklyn Bridge", 1.98, "sans", ""), ("in", 2.71, "sans", ""), ("1883", 2.90, "sans", ""),
    ("so,", 4.15, "sans", ""), ("how did they", 4.52, "sans", ""), ("gain", 5.08, "sans", ""), ("Trust?", 5.34, "script", ""),
    ("instead of", 5.89, "sans", ""), ("going to the", 6.37, "sans", ""), ("national news", 6.95, "sans", ""),
    ("and", 8.01, "sans", ""), ("argue", 8.30, "sans", ""), ("with the", 8.94, "sans", ""), ("people", 9.34, "sans", ""),
    ("why they", 9.96, "sans", ""), ("should", 10.32, "sans", ""), ("trust it", 10.52, "sans", ""), ("they", 11.22, "sans", ""),
    ("let", 11.64, "sans", ""), ("Animals", 12.06, "script", ""), ("cross the", 12.66, "sans", ""), ("bridge", 13.24, "sans", ""),
    ("to", 13.75, "sans", ""), ("explain", 14.05, "sans", ""), ("and", 14.61, "sans", ""), ("showcase", 14.79, "sans", ""),
    ("that it's", 15.77, "sans", ""), ("Trustworthy", 16.19, "script", ""), ("we at", 16.96, "sans", ""),
    ("WKC", 17.57, "script", "PINK"), ("do the same", 18.10, "sans", ""), ("marketing", 18.89, "sans", ""),
    ("is sales", 19.50, "sans", ""), ("we showcase", 20.11, "sans", ""), ("your product", 20.81, "sans", ""),
    ("we showcase", 21.50, "sans", ""), ("why it's", 22.23, "sans", ""), ("trustworthy", 22.57, "sans", ""),
    ("and that's", 23.26, "sans", ""), ("Exactly", 23.74, "script", ""), ("what the", 24.41, "sans", ""),
    ("people", 24.77, "sans", ""), ("are looking for", 25.15, "sans", ""),
]
VIEW_PLAN = [  # (start, view) for the talking head; inserts/canvases cover it where set
    (0.0, "wide"), (4.07, "medium"), (5.34, "close"), (5.89, "wide"), (9.87, "medium"), (11.13, "wide"),
    (13.67, "medium"), (15.67, "close"), (16.90, "wide"), (17.57, "medium"), (18.80, "wide"), (23.20, "wide"),
    (23.74, "medium"), (24.41, "wide"),
]
INSERTS = [  # (start, end, card) - real public-domain prints and newspaper scans (see broll manifests)
    (0.0, 0.50, "ci_fireworks"),        # hook: Currier & Ives, opening night fireworks, 24 May 1883
    (1.10, 2.71, "harper_fireworks"),   # "the opening of the Brooklyn Bridge": Harper's Weekly engraving
    (2.71, 4.07, "ci_date"),            # "in 1883": the print's own caption with the date
    (5.89, 7.93, "sun_page"),           # "the national news": The Sun, 18 May 1884
    (7.93, 9.87, "opper_crowd"),        # "argue with the people": Puck, the grand opening march, 1883
    (12.06, 13.67, "jumbo_headline"),   # "animals cross the bridge": "JUMBO GOES OVER THE BRIDGE."
]
CANVASES = [(18.80, 20.03, "equation"), (20.03, 23.20, "showcase")]
END_CARD = 1.3


def main():
    edl = json.load(open(f"{E}/out/edl.json"))
    speech_end = edl["duration"]
    views = []
    for k, (s, v) in enumerate(VIEW_PLAN):
        e = VIEW_PLAN[k + 1][0] if k + 1 < len(VIEW_PLAN) else speech_end
        views.append(dict(start=s, end=e, view=v))
    inserts = [dict(start=s, end=e, file=f"{E}/broll/v14_{n}.mp4", src=0.0) for s, e, n in INSERTS]
    canvases = [dict(start=s, end=e, kind=k) for s, e, k in CANVASES]
    switches = sorted({round(x, 3) for x in [v["start"] for v in views] + [i["start"] for i in inserts] + [i["end"] for i in inserts]
                       + [c["start"] for c in canvases] + [canvases[-1]["end"], speech_end]})
    neg = set()
    for s in switches:
        f = int(round(s * FPS))
        neg.update(range(f, f + (2 if s in (0.0, 17.57, round(speech_end, 3)) else 1)))
    caps = [dict(text=t, start=s, cls=c, fill=f) for t, s, c, f in CAPTIONS]
    cfg = dict(views=views, inserts=inserts, canvases=canvases, captions=caps, negative_frames=sorted(neg),
               end_card_s=END_CARD, website=None, logo=None)
    web = f"{E}/broll/wkc/manifest.json"
    if os.path.exists(web):
        m = json.load(open(web))
        items = m if isinstance(m, list) else m.get("items", [])
        scroll = [it for it in items if "scroll" in json.dumps(it).lower() and str(it.get("file_vertical", "")).endswith(".mp4")]
        if scroll:
            cfg["website"] = scroll[0]["file_vertical"]
    json.dump(cfg, open(f"{E}/out/edit_v14.json", "w"), indent=1)

    # sound events
    cap_frames = [max(0, int(-(-s * FPS // 1)) - 1) for _, s, _, _ in CAPTIONS]
    ev = [{"t": f / FPS, "type": "shutter"} for f in cap_frames]
    ev += [{"t": cap_frames[k] / FPS, "type": "hit", "size": 1.2 if CAPTIONS[k][0] in ("WKC", "Exactly") else 0.9}
           for k in range(len(CAPTIONS)) if CAPTIONS[k][2] == "script"]
    ev += [{"t": 0.0, "type": "hook"}]
    ev += [{"t": s, "type": "flash"} for s in switches if 0 < s < speech_end and not any(abs(s - c[0]) < 0.01 for c in CANVASES)]
    ev += [{"t": c["start"], "type": "whoosh"} for c in canvases[:1]] + [{"t": canvases[-1]["end"], "type": "whoosh"}]
    ev += [{"t": cap_frames[6] / FPS, "type": "glitch"}]                                  # "1883"
    ev += [{"t": (int(round(c["start"] * FPS)) + 2) / FPS, "type": "glitch"} for c in canvases]
    ev += [{"t": (int(round(canvases[1]["start"] * FPS)) + 5) / FPS, "type": "click"}]
    wkc = cap_frames[[c[0] for c in CAPTIONS].index("WKC")] / FPS
    ev += [{"t": wkc, "type": "riser", "length": 1.2}]
    ev += [{"t": i["start"], "end": i["end"], "type": "projector"} for i in inserts]
    ev += [{"t": speech_end, "type": "hit", "size": 1.3}, {"t": speech_end, "type": "glitch"}]
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{E}/out/v14_events.json", "w"), indent=1)
    print(f"{len(caps)} captions, {len(inserts)} inserts, {len(canvases)} canvases, {len(neg)} negative frames, {len(ev)} sound events; website={cfg['website']}")


if __name__ == "__main__":
    main()
