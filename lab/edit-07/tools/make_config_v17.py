"""Creative decisions for v1 of the Röntgen 1895 -> WKC edit, written to out/edit_v17.json + out/v17_events.json.

The client filmed at 24 fps, so this edit runs at 24 fps. Times are OUTPUT seconds (after the uniform-pacing cuts
in out/edl.json); captions are tied to word indices on the paced timeline. Archival slots are filled from
out/media.json (verified images chosen per moment).
"""
import json
import os

E = "/home/user/SocialMediaMarketingSkill/lab/edit-07"
FPS = 24

CAPTIONS = [  # (text, index of its first word in edl["words"], class, fill)
    ("In 1895", 0, "sans", ""), ("Wilhelm", 4, "sans", ""), ("Röntgen", 5, "script", ""), ("discovered", 6, "sans", ""),
    ("something", 7, "sans", ""), ("Extraordinary", 8, "script", ""), ("he could", 9, "sans", ""), ("see through", 11, "sans", ""),
    ("human skin", 13, "sans", ""), ("instead of", 15, "sans", ""), ("explaining", 17, "sans", ""), ("his discovery", 18, "sans", ""),
    ("he made", 20, "sans", ""), ("an image", 22, "sans", ""), ("of his", 24, "sans", ""), ("wife's hand", 26, "sans", ""),
    ("and literally", 28, "sans", ""), ("for the", 30, "sans", ""), ("first time", 32, "sans", ""), ("you could see", 34, "sans", ""),
    ("Bones", 37, "script", ""), ("One image", 38, "sans", ""), ("made an", 40, "sans", ""), ("Invisible", 42, "script", ""),
    ("world", 43, "sans", ""), ("Visible", 44, "script", ""), ("and that's", 45, "sans", ""), ("the power", 47, "sans", ""),
    ("of Visualization", 49, "script", "PINK"), ("Don't just", 51, "sans", ""), ("explain it", 53, "sans", ""), ("Show it", 55, "script", ""),
]
VIEW_PLAN = [(0.0, "wide"), (2.51, "wide"), (3.40, "medium"), (4.458, "wide"), (6.375, "medium"), (8.458, "wide"),
             (8.95, "medium"), (16.0, "wide"), (16.78, "close"), (17.18, "medium")]
SLOTS = [  # (start, end, slot): filled from out/media.json
    (0.0, 1.45, "hook"),             # "In 1895": the hook frame
    (1.45, 2.51, "rontgen"),         # "Wilhelm Röntgen": his portrait
    (6.96, 8.458, "paper"),          # "explaining his discovery": his paper, Würzburg, December 1895
    (9.54, 10.26, "wife"),           # "of his wife's": Anna Bertha Röntgen
    (10.26, 12.40, "xray"),          # "hand": THE image, 22 December 1895
    (12.40, 13.375, "xray_bones"),   # "you could see bones": closer on the bones and the ring
]
SCAN = (4.458, 6.375)                # "he could see through human skin": an X-ray scan line sweeps over the speaker
CANVASES = [(13.375, 16.0, "reveal"), (18.292, 20.292, "show")]
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
    if "xray" in media:
        canvases[0]["xray"] = media["xray"]["file"]
    beats = dict(
        reveal_title=int(round(13.375 * FPS)) + 2,
        hidden=frame_of("Invisible"), shown=frame_of("Visible"),
        show_title=int(round(18.292 * FPS)) + 2,
        row=frame_of("explain it") + 2, row_to=frame_of("Show it") + 2,
        scan=[int(round(SCAN[0] * FPS)), int(round(SCAN[1] * FPS))],
    )
    switches = sorted({round(x, 3) for x in [v["start"] for v in views] + [i["start"] for i in inserts] + [i["end"] for i in inserts]
                       + [c["start"] for c in canvases] + [c["end"] for c in canvases] + [SCAN[0], SCAN[1], speech_end]})
    neg = set()
    for s in switches:
        f = int(round(s * FPS))
        neg.update(range(f, f + (2 if s in (0.0, 10.26, round(speech_end, 3)) else 1)))   # the X-ray lands with a 2-frame negative
    cfg = dict(fps=FPS, views=views, inserts=inserts, canvases=canvases, scan=list(SCAN), captions=caps, cap_frames=cap_frame,
               negative_frames=sorted(neg), end_card_s=END_CARD, beats=beats,
               logo="/home/user/SocialMediaMarketingSkill/lab/edit-05/broll/wkc/logo.png")
    json.dump(cfg, open(f"{E}/out/edit_v17.json", "w"), indent=1)

    # sound events (PREFERENCES.md "Sound effects per moment")
    ev = [{"t": f / FPS, "type": "shutter"} for f in cap_frame]
    big = {len(caps) - 1, [c["text"] for c in caps].index("of Visualization"), [c["text"] for c in caps].index("Bones")}
    ev += [{"t": cap_frame[k] / FPS, "type": "hit", "size": 1.2 if k in big else 0.9} for k in range(len(caps)) if caps[k]["cls"] == "script"]
    ev += [{"t": 0.0, "type": "hook"}]
    edges = {round(c["start"], 3) for c in canvases} | {round(c["end"], 3) for c in canvases}
    ev += [{"t": s, "type": "flash"} for s in switches if 0 < s < speech_end and s not in edges]
    ev += [{"t": t, "type": "whoosh"} for t in sorted(edges)]
    ev += [{"t": SCAN[0], "type": "scan", "length": SCAN[1] - SCAN[0]}]                      # the X-ray tube's buzz under the scan
    ev += [{"t": 10.26, "type": "hit", "size": 1.3}]                                           # the X-ray lands
    ev += [{"t": beats["reveal_title"] / FPS, "type": "glitch"}, {"t": beats["show_title"] / FPS, "type": "glitch"}]
    ev += [{"t": (beats["shown"] + 2) / FPS, "type": "click"}, {"t": beats["row"] / FPS, "type": "click"}, {"t": beats["row_to"] / FPS, "type": "click"}]
    ev += [{"t": 10.26, "type": "riser", "length": 0.9}]                                      # builds into the X-ray (ends on it)
    ev += [{"t": speech_end, "type": "hit", "size": 1.3}, {"t": speech_end, "type": "glitch"}]
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{E}/out/v17_events.json", "w"), indent=1)
    print(f"{len(caps)} captions, {len(inserts)}/{len(SLOTS)} slots filled, {len(canvases)} canvases, {len(neg)} negative frames, "
          f"{len(ev)} sound events; beats {beats}")


if __name__ == "__main__":
    main()
