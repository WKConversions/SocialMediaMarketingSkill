"""Creative decisions for v1 of the Montgolfier 1783 -> WKC edit, written to out/edit_v15.json + out/v15_events.json.

Times are OUTPUT seconds (after the uniform-pacing cuts in out/edl.json). Captions are tied to word indices on the
paced timeline, so re-running pacing.py keeps them in sync. Print slots are filled from out/prints.json (the
verified archival images chosen for each moment, written by pick_prints.py).
"""
import json
import os

E = "/home/user/SocialMediaMarketingSkill/lab/edit-05"
FPS = 30

CAPTIONS = [  # (text, index of its first word in edl["words"], class, fill)
    ("In 1783", 0, "sans", ""), ("people saw", 4, "sans", ""), ("the first", 6, "sans", ""), ("hot air", 8, "sans", ""),
    ("Balloon", 10, "script", ""), ("rise", 11, "sans", ""), ("into the", 12, "sans", ""), ("Sky", 14, "script", ""),
    ("The Montgolfier", 15, "sans", ""), ("brothers", 17, "sans", ""), ("didn't just", 18, "sans", ""),
    ("Explain", 20, "script", ""), ("their product", 21, "sans", ""), ("but they also", 23, "sans", ""),
    ("Demonstrated", 26, "script", ""), ("it", 27, "sans", ""),
    ("at WK", 28, "script", "PINK"), ("Conversions", 30, "sans", ""), ("we do", 31, "sans", ""), ("the same", 33, "sans", ""),
    ("Imagine", 35, "script", ""), ("sending an", 36, "sans", ""), ("outreach", 38, "sans", ""), ("message", 39, "sans", ""),
    ("saying", 40, "sans", ""), ("we can do", 41, "sans", ""), ("this,", 44, "sans", ""), ("this,", 45, "sans", ""),
    ("this", 46, "sans", ""), ("and this", 47, "sans", ""), ("for you", 49, "sans", ""),
    ("they have to", 51, "sans", ""), ("Believe", 54, "script", ""), ("you", 55, "sans", ""),
    ("what if", 56, "sans", ""), ("you could", 58, "sans", ""), ("Show", 60, "script", ""), ("them", 61, "sans", ""),
    ("like the", 62, "sans", ""), ("people did", 64, "sans", ""), ("to show the", 66, "sans", ""), ("hot air", 69, "sans", ""),
    ("Balloon", 71, "script", ""), ("would they", 72, "sans", ""), ("Believe", 74, "script", ""), ("you?", 75, "sans", ""),
]
# the talking head (wide = the client's own framing; medium 1.08x / close 1.15x centred crops). Prints and canvases
# cover it where set. Switches sit on the pacing cuts or on emphasis words, so jump cuts are hidden.
VIEW_PLAN = [(0.0, "wide"), (1.367, "wide"), (2.11, "medium"), (5.43, "medium"), (6.51, "close"), (7.14, "wide"),
             (9.167, "medium"), (9.95, "wide"), (10.43, "close"), (15.50, "medium"), (16.24, "close"),
             (20.533, "medium"), (21.00, "close")]
SLOTS = [  # (start, end, slot) - real 1783 prints; the slot is filled from out/prints.json
    (0.0, 1.367, "hook"),          # "In 1783": a print of the 1783 ascents with the date on it, the hook frame
    (2.92, 4.433, "rise"),         # "balloon rise into the sky": the balloon rising, panned upwards
    (4.433, 5.43, "brothers"),     # "The Montgolfier": the brothers (v1: no portrait yet -> the talking head)
    (7.70, 9.167, "demo"),         # "but they also demonstrated it": a public demonstration (Annonay / Versailles)
    (18.10, 19.29, "crowd"),       # "like the people did": the crowd watching
    (19.29, 20.533, "ascent"),     # "to show the hot air balloon": the balloon going up before the crowd
]
CANVASES = [(11.10, 15.50, "outreach"), (16.80, 18.10, "showcase")]
END_CARD = 1.3


def main():
    edl = json.load(open(f"{E}/out/edl.json"))
    words = edl["words"]
    speech_end = edl["duration"]
    caps = []
    for text, wi, cls, fill in CAPTIONS:
        caps.append(dict(text=text, start=words[wi]["start"], cls=cls, fill=fill, word=words[wi]["word"]))
    views = []
    for k, (s, v) in enumerate(VIEW_PLAN):
        e = VIEW_PLAN[k + 1][0] if k + 1 < len(VIEW_PLAN) else speech_end
        views.append(dict(start=s, end=e, view=v))
    prints = json.load(open(f"{E}/out/prints.json")) if os.path.exists(f"{E}/out/prints.json") else {}
    inserts = [dict(start=s, end=e, slot=n, **prints[n]) for s, e, n in SLOTS if n in prints]
    canvases = [dict(start=s, end=e, kind=k) for s, e, k in CANVASES]
    wkc = json.load(open(f"{E}/broll/wkc/manifest.json")) if os.path.exists(f"{E}/broll/wkc/manifest.json") else {}
    cap_frame = [max(0, int(-(-c["start"] * FPS // 1)) - 1) for c in caps]

    def frame_of(text, nth=0):
        return [f for c, f in zip(caps, cap_frame) if c["text"] == text][nth]

    # canvas beats (frames): the outreach card fills in on the words that name it
    beats = dict(
        outreach_title=int(round(11.10 * FPS)) + 2,
        bubble=frame_of("saying") + 2,
        rows=[frame_of("this,", 0) + 2, frame_of("this,", 1) + 2, frame_of("this") + 2, frame_of("and this") + 4],
        sent=frame_of("for you") + 2,
        showcase_title=int(round(16.80 * FPS)) + 2,
    )
    switches = sorted({round(x, 3) for x in [v["start"] for v in views] + [i["start"] for i in inserts] + [i["end"] for i in inserts]
                       + [c["start"] for c in canvases] + [c["end"] for c in canvases] + [speech_end]})
    neg = set()
    for s in switches:
        f = int(round(s * FPS))
        neg.update(range(f, f + (2 if s in (0.0, 9.167, round(speech_end, 3)) else 1)))
    # "we can do this, this, this and this": the four things the site says WKC makes (wkconversions.com hero:
    # "website hero explainers, service and product explainers, social media content and social ads")
    services = ["Hero explainers", "Product explainers", "Social content", "Social ads"]
    cfg = dict(views=views, inserts=inserts, canvases=canvases, captions=caps, cap_frames=cap_frame,
               negative_frames=sorted(neg), end_card_s=END_CARD, beats=beats, services=services[:4],
               website=wkc.get("scroll_video") or f"{E}/broll/wkc/scroll_v15.mp4",
               logo=wkc.get("logo_png") or f"{E}/broll/wkc/logo.png",
               logo_on_dark_ok=wkc.get("logo_on_dark_ok", True), brand_colours=wkc.get("brand_colours") or [])
    json.dump(cfg, open(f"{E}/out/edit_v15.json", "w"), indent=1)

    # sound events (rules: PREFERENCES.md "Sound effects per moment")
    ev = [{"t": f / FPS, "type": "shutter"} for f in cap_frame]
    big = {[c["text"] for c in caps].index("at WK"), len(caps) - 2}                       # "at WK" and the final "Believe"
    ev += [{"t": cap_frame[k] / FPS, "type": "hit", "size": 1.2 if k in big else 0.9}
           for k in range(len(caps)) if caps[k]["cls"] == "script"]
    ev += [{"t": 0.0, "type": "hook"}]
    canvas_edges = {round(c["start"], 3) for c in canvases} | {round(c["end"], 3) for c in canvases}
    ev += [{"t": s, "type": "flash"} for s in switches if 0 < s < speech_end and s not in canvas_edges]
    ev += [{"t": t, "type": "whoosh"} for t in sorted(canvas_edges)]
    ev += [{"t": (int(round(i["start"] * FPS))) / FPS, "type": "whoosh"} for i in inserts if i["slot"] == "rise"]
    ev += [{"t": beats["outreach_title"] / FPS, "type": "glitch"}, {"t": beats["showcase_title"] / FPS, "type": "glitch"}]
    ev += [{"t": f / FPS, "type": "click"} for f in beats["rows"] + [beats["bubble"], beats["sent"]]]
    ev += [{"t": cap_frame[[c["text"] for c in caps].index("at WK")] / FPS, "type": "riser", "length": 1.2}]
    ev += [{"t": frame_of("Believe", 1) / FPS, "type": "riser", "length": 0.8}]
    ev += [{"t": speech_end, "type": "hit", "size": 1.3}, {"t": speech_end, "type": "glitch"}]
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{E}/out/v15_events.json", "w"), indent=1)
    print(f"{len(caps)} captions, {len(inserts)}/{len(SLOTS)} print slots filled, {len(canvases)} canvases, "
          f"{len(neg)} negative frames, {len(ev)} sound events; services={cfg['services']}; website={cfg['website']}")


if __name__ == "__main__":
    main()
