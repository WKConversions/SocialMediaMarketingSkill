"""Creative decisions for v1 of the Galileo 1610 -> WKC edit, written to out/edit_v16.json + out/v16_events.json.

Times are OUTPUT seconds (after the uniform-pacing cuts in out/edl.json); captions are tied to word indices on the
paced timeline. Archival slots are filled from out/media.json (verified images / NASA media chosen per moment).
"""
import json
import os

E = "/home/user/SocialMediaMarketingSkill/lab/edit-06"
FPS = 30

CAPTIONS = [  # (text, index of its first word in edl["words"], class, fill)
    ("In 1610", 0, "sans", ""), ("Galileo", 3, "script", ""), ("pointed his", 4, "sans", ""), ("telescope", 6, "sans", ""),
    ("to", 7, "sans", ""), ("Jupiter", 8, "script", ""), ("he discovered", 9, "sans", ""), ("four moons", 11, "sans", ""),
    ("orbiting", 13, "sans", ""), ("another planet", 14, "sans", ""), ("and this", 16, "sans", ""), ("discovery", 18, "sans", ""),
    ("has challenged", 19, "sans", ""), ("Centuries", 21, "script", ""), ("of disbelief", 22, "sans", ""),
    ("he didn't just", 24, "sans", ""), ("make a claim", 27, "sans", ""), ("he had", 30, "sans", ""),
    ("something that", 32, "sans", ""), ("people could", 34, "sans", ""), ("Observe", 36, "script", ""),
    ("and that", 37, "sans", ""), ("is the", 39, "sans", ""), ("Power", 41, "script", ""), ("of Visualization", 42, "script", "PINK"),
    ("make the", 44, "sans", ""), ("complicated", 46, "sans", ""), ("Understandable", 47, "script", ""), ("and the", 48, "sans", ""),
    ("unbelievable", 50, "sans", ""), ("easier to", 51, "sans", ""), ("Believe", 53, "script", ""),
]
VIEW_PLAN = [(0.0, "wide"), (1.75, "wide"), (2.25, "medium"), (6.30, "medium"), (6.72, "wide"), (7.68, "close"),
             (9.867, "medium"), (10.63, "wide"), (13.433, "wide"), (14.73, "close"), (15.37, "medium")]
SLOTS = [  # (start, end, slot): filled from out/media.json
    (0.0, 1.10, "hook"),            # "In 1610": Sidereus Nuncius (1610), the hook frame
    (1.10, 1.75, "galileo"),        # "Galileo": portrait
    (8.18, 9.867, "geocentric"),    # "centuries of disbelief": the geocentric universe
    (11.333, 12.42, "demonstration"),   # "he had something": Galileo showing the telescope
    (12.42, 13.433, "observation"),     # "that people could observe": his 1610 sketches of Jupiter's moons
]
CANVASES = [(3.05, 6.30, "moons"), (16.467, 21.333, "transform")]   # moons: from "Jupiter"
END_CARD = 1.3


def main():
    edl = json.load(open(f"{E}/out/edl.json"))
    words = edl["words"]
    speech_end = edl["duration"]
    caps = [dict(text=t, start=words[wi]["start"], cls=c, fill=f, word=words[wi]["word"]) for t, wi, c, f in CAPTIONS]
    cap_frame = [max(0, int(-(-c["start"] * FPS // 1)) - 1) for c in caps]

    def frame_of(text, nth=0):
        return [f for c, f in zip(caps, cap_frame) if c["text"] == text][nth]

    views = []
    for k, (s, v) in enumerate(VIEW_PLAN):
        e = VIEW_PLAN[k + 1][0] if k + 1 < len(VIEW_PLAN) else speech_end
        views.append(dict(start=s, end=e, view=v))
    media = json.load(open(f"{E}/out/media.json")) if os.path.exists(f"{E}/out/media.json") else {}
    inserts = [dict(start=s, end=e, slot=n, **media[n]) for s, e, n in SLOTS if n in media]
    canvases = [dict(start=s, end=e, kind=k) for s, e, k in CANVASES]
    if "jupiter_disc" in media:
        canvases[0]["jupiter"] = media["jupiter_disc"]["file"]
        canvases[0]["jupiter_circle"] = media["jupiter_disc"].get("circle")
    beats = dict(
        moons_title=int(round(3.05 * FPS)) + 2,
        moons=[frame_of("four moons") + 2 + k * 3 for k in range(4)],         # pop one by one on "four moons" (2 frames after its shutter)
        orbit=frame_of("orbiting"),                                           # they start to move on "orbiting"
        transform_title=int(round(16.467 * FPS)) + 2,
        row1=frame_of("complicated") + 2, row1_to=frame_of("Understandable") + 2,
        row2=frame_of("unbelievable") + 2, row2_to=frame_of("Believe") + 2,
    )
    switches = sorted({round(x, 3) for x in [v["start"] for v in views] + [i["start"] for i in inserts] + [i["end"] for i in inserts]
                       + [c["start"] for c in canvases] + [c["end"] for c in canvases] + [speech_end]})
    neg = set()
    for s in switches:
        f = int(round(s * FPS))
        neg.update(range(f, f + (2 if s in (0.0, 13.433, round(speech_end, 3)) else 1)))
    cfg = dict(views=views, inserts=inserts, canvases=canvases, captions=caps, cap_frames=cap_frame, negative_frames=sorted(neg),
               end_card_s=END_CARD, beats=beats, logo="/home/user/SocialMediaMarketingSkill/lab/edit-05/broll/wkc/logo.png")
    json.dump(cfg, open(f"{E}/out/edit_v16.json", "w"), indent=1)

    # sound events (PREFERENCES.md "Sound effects per moment")
    ev = [{"t": f / FPS, "type": "shutter"} for f in cap_frame]
    big = {len(caps) - 1, [c["text"] for c in caps].index("of Visualization")}
    ev += [{"t": cap_frame[k] / FPS, "type": "hit", "size": 1.2 if k in big else 0.9} for k in range(len(caps)) if caps[k]["cls"] == "script"]
    ev += [{"t": 0.0, "type": "hook"}]
    edges = {round(c["start"], 3) for c in canvases} | {round(c["end"], 3) for c in canvases}
    ev += [{"t": s, "type": "flash"} for s in switches if 0 < s < speech_end and s not in edges]
    ev += [{"t": t, "type": "whoosh"} for t in sorted(edges)]
    ev += [{"t": beats["moons_title"] / FPS, "type": "glitch"}, {"t": beats["transform_title"] / FPS, "type": "glitch"}]
    ev += [{"t": f / FPS, "type": "click"} for f in beats["moons"] + [beats["row1"], beats["row1_to"], beats["row2"], beats["row2_to"]]]
    ev += [{"t": frame_of("Power") / FPS, "type": "riser", "length": 1.0}]
    ev += [{"t": speech_end, "type": "hit", "size": 1.3}, {"t": speech_end, "type": "glitch"}]
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{E}/out/v16_events.json", "w"), indent=1)
    print(f"{len(caps)} captions, {len(inserts)}/{len(SLOTS)} slots filled, {len(canvases)} canvases, {len(neg)} negative frames, "
          f"{len(ev)} sound events; beats {beats}")


if __name__ == "__main__":
    main()
