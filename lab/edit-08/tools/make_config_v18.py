"""Creative decisions for v1 of the Savery 1698 -> WKC edit (24 fps), written to out/edit_v18.json + out/v18_events.json.

Times are OUTPUT seconds (after the uniform-pacing cuts in out/edl.json); captions are tied to word indices on the
paced timeline. Archival slots are filled from out/media.json (verified images chosen per moment).
"""
import json
import os

E = "/home/user/SocialMediaMarketingSkill/lab/edit-08"
FPS = 24

CAPTIONS = [  # (text, index of its first word in edl["words"], class, fill)
    ("In 1698", 0, "sans", ""), ("Thomas", 4, "sans", ""), ("Savery", 5, "script", ""), ("invented", 6, "sans", ""),
    ("a machine", 7, "sans", ""), ("that's powered", 9, "sans", ""), ("by Steam", 11, "script", ""),
    ("at the time", 13, "sans", ""), ("the idea", 16, "sans", ""), ("was very", 18, "sans", ""), ("hard to", 20, "sans", ""),
    ("Imagine", 22, "script", ""), ("so he first", 23, "sans", ""), ("Demonstrated", 26, "script", ""), ("how steam", 27, "sans", ""),
    ("could move", 29, "sans", ""), ("water", 31, "sans", ""), ("and suddenly", 32, "sans", ""), ("the invention", 34, "sans", ""),
    ("had a", 36, "sans", ""), ("Purpose", 38, "script", ""), ("and that's", 39, "sans", ""), ("exactly what", 41, "sans", ""),
    ("Motion Design", 43, "script", "PINK"), ("does for", 45, "sans", ""), ("businesses", 47, "sans", ""),
    ("it makes", 48, "sans", ""), ("Complexity", 50, "script", ""), ("easier to", 51, "sans", ""), ("Understand", 53, "script", ""),
]
VIEW_PLAN = [(0.0, "wide"), (2.73, "wide"), (3.43, "medium"), (5.167, "wide"), (6.73, "medium"), (7.56, "close"),
             (13.708, "wide"), (15.02, "close"), (15.70, "medium")]
SLOTS = [  # (start, end, slot): filled from out/media.json
    (0.0, 1.68, "hook"),            # "In 1698": the engine engraving (The Miner's Friend, 1702)
    (1.68, 2.73, "title"),          # "Thomas Savery": the title page with his name (no authentic portrait is known)
    (4.11, 5.167, "philtrans"),     # "powered by steam": the 1699 Royal Society figure of his engine
]
CANVASES = [(8.042, 13.708, "engine"), (16.708, 19.292, "untangle")]
END_CARD = 1.3


def main():
    edl = json.load(open(f"{E}/out/edl.json"))
    words = edl["words"]
    speech_end = edl["duration"]
    caps = [dict(text=t, start=words[wi]["start"], cls=c, fill=f, word=words[wi]["word"]) for t, wi, c, f in CAPTIONS]
    cap_frame = [max(0, int(-(-c["start"] * FPS // 1)) - 1) for c in caps]

    def frame_of(text, nth=0):
        return [f for c, f in zip(caps, cap_frame) if c["text"] == text][nth]

    def word_frame(i):
        return int(round(words[i]["start"] * FPS))

    views = []
    for k, (s, v) in enumerate(VIEW_PLAN):
        e = VIEW_PLAN[k + 1][0] if k + 1 < len(VIEW_PLAN) else speech_end
        views.append(dict(start=s, end=e, view=v))
    media = json.load(open(f"{E}/out/media.json")) if os.path.exists(f"{E}/out/media.json") else {}
    inserts = [dict(start=s, end=e, slot=n, **media[n]) for s, e, n in SLOTS if n in media]
    canvases = [dict(start=s, end=e, kind=k) for s, e, k in CANVASES]
    # the engine animation follows the words: draw on "demonstrated", steam on "steam", vacuum + water up on "move",
    # water pushed up and out on "water", the purpose label on "purpose"
    beats = dict(
        engine_title=int(round(8.042 * FPS)) + 2,
        draw=word_frame(26), steam=word_frame(28), condense=word_frame(30) - 4, rise=word_frame(30), push=word_frame(31),
        purpose=frame_of("Purpose") + 2,
        untangle_title=int(round(16.708 * FPS)) + 2, tangle=frame_of("Complexity") + 2, straight=word_frame(53),
    )
    switches = sorted({round(x, 3) for x in [v["start"] for v in views] + [i["start"] for i in inserts] + [i["end"] for i in inserts]
                       + [c["start"] for c in canvases] + [c["end"] for c in canvases] + [speech_end]})
    neg = set()
    for s in switches:
        f = int(round(s * FPS))
        neg.update(range(f, f + (2 if s in (0.0, round(speech_end, 3)) else 1)))
    cfg = dict(fps=FPS, views=views, inserts=inserts, canvases=canvases, captions=caps, cap_frames=cap_frame,
               negative_frames=sorted(neg), end_card_s=END_CARD, beats=beats,
               logo="/home/user/SocialMediaMarketingSkill/lab/edit-05/broll/wkc/logo.png")
    json.dump(cfg, open(f"{E}/out/edit_v18.json", "w"), indent=1)

    # sound events (PREFERENCES.md "Sound effects per moment")
    ev = [{"t": f / FPS, "type": "shutter"} for f in cap_frame]
    big = {len(caps) - 1, [c["text"] for c in caps].index("Motion Design")}
    ev += [{"t": cap_frame[k] / FPS, "type": "hit", "size": 1.2 if k in big else 0.9} for k in range(len(caps)) if caps[k]["cls"] == "script"]
    ev += [{"t": 0.0, "type": "hook"}]
    edges = {round(c["start"], 3) for c in canvases} | {round(c["end"], 3) for c in canvases}
    ev += [{"t": s, "type": "flash"} for s in switches if 0 < s < speech_end and s not in edges]
    ev += [{"t": t, "type": "whoosh"} for t in sorted(edges)]
    ev += [{"t": beats["engine_title"] / FPS, "type": "glitch"}, {"t": beats["untangle_title"] / FPS, "type": "glitch"}]
    ev += [{"t": beats["steam"] / FPS, "type": "hiss", "length": 0.7}, {"t": beats["push"] / FPS, "type": "hiss", "length": 0.9}]
    ev += [{"t": f / FPS, "type": "click"} for f in (beats["draw"] + 2, beats["purpose"], beats["straight"] + 2)]
    ev += [{"t": frame_of("Motion Design") / FPS, "type": "riser", "length": 0.9}]
    ev += [{"t": speech_end, "type": "hit", "size": 1.3}, {"t": speech_end, "type": "glitch"}]
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{E}/out/v18_events.json", "w"), indent=1)
    print(f"{len(caps)} captions, {len(inserts)}/{len(SLOTS)} slots filled, {len(canvases)} canvases, {len(neg)} negative frames, "
          f"{len(ev)} sound events; beats {beats}")


if __name__ == "__main__":
    main()
