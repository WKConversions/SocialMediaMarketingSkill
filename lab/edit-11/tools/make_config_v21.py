"""Creative decisions for v1 of the Edward Johnson 1882 Christmas tree -> WKC edit (24 fps), the classy / jazz style.

The client's brief: the same style as the Wright and Eiffel videos; Sinatra "In the Wee Small Hours of the Morning"
from the moment he sings "In the wee small..." (attack at 16.23 s in the file, start 16.20); very cozy and warm;
music-only fragments between his explaining and warm images, some under his voice; the majority him explaining; no
change to the colour or lighting of his footage; a soft shutter on every caption.

Song landmarks (video time = song time - 16.20): "In the wee small hours of the morning" 0.03-~6.2, "While the whole wide
world is fast asleep" 7.5-13.5, a vocal pause 13.5-15.8, "You lie awake..." from 15.8.
The paced voice (out/edl.json, 19.375 s) is split into three blocks at sentence pauses. All times are VIDEO seconds
unless named voice time. Visual slots are filled from out/media.json (fact-checked, commercially cleared items only).
Writes out/edit_v21.json + out/v21_events.json.
"""
import json
import os

E = "/home/user/SocialMediaMarketingSkill/lab/edit-11"
FPS = 24
SONG_START_S = 16.20                    # music/library.json "sinatra-in-the-wee-small-hours" start_offset_s

# (voice start, voice end, video start frame): block boundaries sit in the pacing pauses between sentences
BLOCKS = [(0.0, 7.0, 150),              # "In 1882 ... candles. But then, Edward Johnson had a different idea." (after "...morning")
          (7.0, 13.6667, 357),          # "He decorated a Christmas tree with 80 electric lights. Rather than ... he showed it."
          (13.6667, 19.375, 565)]       # "A simple demonstration ... inspiring Christmas. So, some ideas speak for themselves."
END_CARD_S = 1.75

# interludes (music up, visuals): start/end are filled from the blocks; slot sequence with relative weights
INTERLUDES = [
    dict(name="I0", slots=[("cozy_a", 1.0), ("cozy_b", 1.0)]),       # Sinatra's first line: warm period Christmas
    dict(name="I1", slots=[("lamp", 1.0)]),                           # "...a different idea": the electric lamp (in his pause)
    dict(name="I2", slots=[("electric", 1.0)]),                       # "...he showed it": light on a Christmas tree
]
# visuals under his voice (voice time)
UNDER_VOICE = [
    (1.40, 4.375, "candles"),           # "Christmas trees were still lighted up by candles": a candle-lit tree
    (9.00, 10.5417, "lights"),          # "with 80 electric lights"
]
# speaker framing per paced segment (index into edl["segments"]): every jump cut changes the framing
VIEWS = ["wide", "medium", "wide", "medium", "wide", "medium", "wide", "medium", "wide"]

CAPTIONS = [  # (text, index of its first word in edl["words"]): what he said, phrase by phrase
    ("In 1882", 0), ("Christmas trees were", 4), ("still lighted up by candles", 7), ("But then", 12),
    ("Edward Johnson", 14), ("had a different idea", 16), ("He decorated", 20), ("a Christmas tree", 22),
    ("with 80 electric lights", 25), ("Rather than explaining", 29), ("the potential of the product", 32),
    ("he showed it", 37), ("A simple demonstration", 40), ("that helped a lasting", 43), ("inspiring Christmas", 47),
    ("So, some ideas", 49), ("speak for themselves", 52),
]
DISSOLVE = dict(interlude=0.6, under_voice=0.4, within=0.5)   # cross-dissolve lengths (s)
MUSIC = dict(down_lead_s=0.30, down_len_s=0.42, up_after_s=0.06, up_len_s=0.40, fade_out_s=1.6)


def main():
    edl = json.load(open(f"{E}/out/edl.json"))
    words, segs = edl["words"], edl["segments"]

    def v2t(vt, end=False):
        """voice time -> video time (a block edge belongs to the block it starts, or with end=True the one it ends)"""
        for v0, v1, f0 in BLOCKS:
            if (v0 - 1e-6 < vt <= v1 + 1e-6) if end else (v0 - 1e-6 <= vt < v1 - 1e-6):
                return round(f0 / FPS + vt - v0, 4)
        raise ValueError(vt)

    blocks = [dict(v0=v0, v1=v1, start=round(f0 / FPS, 4), end=round(f0 / FPS + v1 - v0, 4)) for v0, v1, f0 in BLOCKS]
    for b in blocks:
        assert abs(b["end"] * FPS - round(b["end"] * FPS)) < 5e-3, b              # edl times are rounded to 0.1 ms
    total = blocks[-1]["end"] + END_CARD_S
    media = json.load(open(f"{E}/out/media.json")) if os.path.exists(f"{E}/out/media.json") else {}

    # interludes fill the gaps before each block
    gaps = [(0.0, blocks[0]["start"])] + [(blocks[k - 1]["end"], blocks[k]["start"]) for k in range(1, len(blocks))]
    shots = []
    for il, (s, e) in zip(INTERLUDES, gaps):
        t = s
        wsum = sum(w for _, w in il["slots"])
        for name, w in il["slots"]:
            d = (e - s) * w / wsum
            shots.append(dict(start=round(t, 4), end=round(t + d, 4), slot=name, part=il["name"]))
            t += d
    for v0, v1, name in UNDER_VOICE:
        shots.append(dict(start=v2t(v0), end=v2t(v1, end=True), slot=name, part="voice"))
    shots.sort(key=lambda x: x["start"])
    # merge touching shots of the same slot (the photograph carried from I2 into "A machine lifted off the ground")
    merged = []
    for sh in shots:
        if merged and merged[-1]["slot"] == sh["slot"] and sh["start"] <= merged[-1]["end"] + 1e-3:
            merged[-1]["end"] = max(merged[-1]["end"], sh["end"])
            merged[-1]["part"] += "+" + sh["part"]
        else:
            merged.append(dict(sh))
    shots = []
    for sh in merged:
        key = "photo" if sh["slot"] == "photo_callback" else sh["slot"]
        if key in media:
            item = media[key]
            sh.update({k: v for k, v in item.items() if k not in ("start", "end")})
            if sh["slot"] == "photo_callback":          # the callback: tighter on the machine in the air
                sh.update(item.get("callback", {}))
        shots.append(sh)

    # speaker segments on the video timeline
    speaker = []
    for k, sg in enumerate(segs):
        speaker.append(dict(clip=sg["clip"], src_start=sg["src_start"], src_end=sg["src_end"],
                            start=v2t(sg["out_start"]),
                            end=v2t(sg["out_end"], end=True), view=VIEWS[k]))

    caps = []
    for k, (text, wi) in enumerate(CAPTIONS):
        t = v2t(words[wi]["start"])
        f = max(0, int(-(-t * FPS // 1)) - 1)                          # one frame before the word
        caps.append(dict(text=text, word=words[wi]["word"], word_t=t, frame=f))
    for k, c in enumerate(caps):                                         # each caption holds until the next, or its block's end
        blk = next(b for b in blocks if b["start"] - 1e-6 <= c["word_t"] <= b["end"] + 1e-6)
        nxt = caps[k + 1]["frame"] if k + 1 < len(caps) and caps[k + 1]["word_t"] < blk["end"] else None
        c["end_frame"] = nxt if nxt is not None else int(round(blk["end"] * FPS)) + 6
        c["fade_out"] = nxt is None
    # place labels in the interludes (same font, same spot); text comes from the fact-checked media record
    labels = []
    for sh in shots:
        if sh.get("label") and sh["part"].startswith("I"):
            f0 = int(round(sh["start"] * FPS)) + (8 if sh["start"] > 0 else 3)
            end = sh["end"]
            if sh.get("label_spans_part"):                                    # one label over a whole montage
                end = max(x["end"] for x in shots if x["part"] == sh["part"])
            labels.append(dict(text=sh["label"], frame=f0, end_frame=int(round(end * FPS)) - 4, fade_out=True, label=True))
    # a move shared by a montage: its span is the montage's time range
    for sh in shots:
        if sh.get("push_span") == "part":
            sh["push_span"] = [min(x["start"] for x in shots if x["part"] == sh["part"]),
                               max(x["end"] for x in shots if x["part"] == sh["part"])]

    # music automation: up in the interludes and end card, down under his voice, smooth ramps
    first_w = [v2t(next(w for w in words if w["start"] >= b["v0"] - 1e-6)["start"]) for b in blocks]
    last_w = [v2t(max((w for w in words if w["end"] <= b["v1"] + 1e-6), key=lambda w: w["end"])["end"], end=True) for b in blocks]
    ramps = []
    for fw, lw in zip(first_w, last_w):
        d0 = fw - MUSIC["down_lead_s"]
        ramps.append(dict(t0=round(d0, 4), t1=round(d0 + MUSIC["down_len_s"], 4), to="voice"))
        u0 = lw + MUSIC["up_after_s"]
        ramps.append(dict(t0=round(u0, 4), t1=round(u0 + MUSIC["up_len_s"], 4), to="up"))

    cfg = dict(fps=FPS, duration=round(total, 4), song_start_s=SONG_START_S, blocks=blocks, speaker=speaker, shots=shots,
               captions=caps, labels=labels, dissolve=DISSOLVE, music_ramps=ramps, music=MUSIC,
               end_card=dict(start=blocks[-1]["end"], text=["Some ideas speak", "for themselves"]),
               logo="/home/user/SocialMediaMarketingSkill/lab/edit-05/broll/wkc/logo.png")
    json.dump(cfg, open(f"{E}/out/edit_v21.json", "w"), indent=1)

    ev = [{"t": c["frame"] / FPS, "type": "shutter"} for c in caps + labels]
    ev += [{"t": round(cfg["end_card"]["start"] + 0.35, 4), "type": "shutter"}]
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{E}/out/v21_events.json", "w"), indent=1)

    # balance report (client: ~50% explaining, ~30% music + visuals, ~20% visuals under the voice)
    voice_total = sum(b["v1"] - b["v0"] for b in blocks)
    under = sum(min(sh["end"], b["end"]) - max(sh["start"], b["start"]) for sh in shots for b in blocks
                if min(sh["end"], b["end"]) > max(sh["start"], b["start"]))
    music_part = total - voice_total
    print(f"{total:.2f} s: explaining on camera {voice_total - under:.2f} s ({(voice_total - under) / total:.0%}), "
          f"music + visuals {music_part:.2f} s ({music_part / total:.0%}), visuals under the voice {under:.2f} s ({under / total:.0%})")
    print(f"{len(caps)} captions, {len(labels)} labels, {len(shots)} shots ({sum(1 for s in shots if 'file' in s)} filled), "
          f"song {SONG_START_S}-{SONG_START_S + total:.2f} s")
    for b, fw, lw in zip(blocks, first_w, last_w):
        print(f"  block {b['start']:6.3f}-{b['end']:6.3f}  first word {fw:6.3f}  last word ends {lw:6.3f}")
    for sh in shots:
        print(f"  shot {sh['start']:6.3f}-{sh['end']:6.3f} {sh['slot']:15s} {sh['part']}")


if __name__ == "__main__":
    main()
