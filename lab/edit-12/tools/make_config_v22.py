"""Creative decisions for v1 of the Henry Cole 1843 first Christmas card -> WKC edit (24 fps), the classy / jazz style.

The client's brief: the same style as the earlier videos; Sinatra "It Had To Be You" from half a second before the
very first "It had to be you" (46.30 s in the file -> start 45.80); no change to the lighting of his footage; a longer
video, so the music + visual moments are a little longer, with the same shares (~50% explaining, ~30% music + visuals,
~20% visuals under his voice); a soft shutter on every caption.

Song landmarks (video time = song time - 45.80): "It had to be you" 0.5-4.6, again 8.8-12, "I wandered around..."
~16-22, big-band fills between lines up to ~44, then the long sung middle section.
The paced voice (out/edl.json, 63.458 s) is split into six blocks at paragraph pauses. All times are VIDEO seconds
unless named voice time. Visual slots are filled from out/media.json (fact-checked, commercially cleared items only).
Writes out/edit_v22.json + out/v22_events.json.
"""
import json
import os

E = "/home/user/SocialMediaMarketingSkill/lab/edit-12"
FPS = 24
SONG_START_S = 45.80                    # music/library.json "sinatra-it-had-to-be-you" start_offset_s

# (voice start, voice end, video start frame): block boundaries sit in the pacing pauses between paragraphs
BLOCKS = [(0.0, 12.25, 114),            # "In the winter of 1843 ... His name was Sir Henry Cole."  (after "It had to be you")
          (12.25, 25.6667, 486),        # "He had a rather simple problem ... came up with a lovely idea."
          (25.6667, 32.5, 886),         # "Instead of writing a different letter ... warmth of Christmas?"
          (32.5, 42.9167, 1146),        # "Horsley painted a family gathering ... A Merry Christmas and a Happy New Year to you."
          (42.9167, 51.8333, 1474),     # "It was a small gesture ... a tradition for millions of people around the world."
          (51.8333, 63.4583, 1784)]     # "And perhaps that's what makes it so special ... a thousand words cannot say."
END_CARD_S = 2.75

# interludes (music up, visuals): start/end are filled from the blocks; slot sequence with relative weights
INTERLUDES = [
    dict(name="I0", slots=[("london", 1.0)]),                          # "It had to be you": London in winter
    dict(name="I1", slots=[("letters", 1.0)]),                         # after "Sir Henry Cole": letters and the post
    dict(name="I2", slots=[("cozy", 1.0)]),                            # after "a lovely idea": a warm Victorian Christmas
    dict(name="I3", slots=[("card", 1.0)]),                            # after "...warmth of Christmas?": THE 1843 card
    dict(name="I4", slots=[("victorian", 1.0)]),                       # after "...Happy New Year to you": Christmas spreading
    dict(name="I5", slots=[("cards_a", 1.0), ("cards_b", 1.0)]),      # after "...around the world": later Christmas cards
]
# visuals under his voice (voice time)
UNDER_VOICE = [
    (0.0, 2.0, "london"),               # "In the winter of 1843": the opening image stays
    (10.4167, 12.25, "cole"),           # "His name was Sir Henry Cole": his portrait
    (20.4167, 23.2083, "horsley"),      # "an artist named John Callcott Horsley": his portrait
    (30.00, 32.5, "card"),              # "why not send an illustration carrying the warmth of Christmas": the card appears
    (32.5, 34.2083, "card_family"),     # "Horsley painted a family gathering": the centre panel
    (34.2083, 37.5, "card_kindness"),   # "with a lot of kindness and warmth from either side": a side panel
    (40.5, 42.9167, "card_words"),      # "A Merry Christmas and a Happy New Year to you": the words on the card
    (49.00, 51.8333, "cards_a"),        # "a tradition for millions of people around the world": a later card
]
# speaker framing per paced segment (index into edl["segments"]): every jump cut changes the framing
VIEWS = ["wide", "medium"] * 15

CAPTIONS = [  # (text, index of its first word in edl["words"]): what he said, phrase by phrase
    ("In the winter of 1843", 0), ("the London streets were filled", 7), ("with the charm of Christmas", 12),
    ("Behind the desk sat a gentleman", 17), ("surrounded by envelopes", 23), ("and letters", 26),
    ("His name was", 28), ("Sir Henry Cole", 31),
    ("He had a rather simple problem", 34), ("There are so many people", 40), ("he wanted to wish", 45),
    ("a Merry Christmas", 49), ("but he had so little time", 52), ("to write them all", 58),
    ("So he turned to his friend", 62), ("an artist named", 68), ("John Callcott Horsley", 71),
    ("Together they came up", 74), ("with a lovely idea", 78),
    ("Instead of writing", 82), ("a different letter", 85), ("to every single friend", 88),
    ("why not send an illustration", 92), ("carrying the warmth", 97), ("of Christmas", 100),
    ("Horsley painted", 102), ("a family gathering", 104), ("with a lot of kindness", 107), ("and warmth", 112),
    ("from either side", 114),
    ("And beneath the illustration", 117), ("were the written words", 121),
    ("A Merry Christmas", 125), ("and a Happy New Year", 128), ("to you", 133),
    ("It was a small gesture", 135), ("a piece of paper", 140), ("a small illustration", 144), ("and a few words", 147),
    ("Yet that gesture became", 151), ("a tradition for millions", 155), ("of people around the world", 159),
    ("And perhaps that's what", 164), ("makes it so special", 168),
    ("Sometimes the most", 172), ("meaningful messages", 175), ("aren't the longest", 177),
    ("They are the ones", 180), ("that make us feel something", 184),
    ("Because a picture can say", 189), ("what a thousand words", 194), ("cannot say", 198),
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
               end_card=dict(start=blocks[-1]["end"], text=["A picture can say", "what a thousand", "words cannot"]),
               logo="/home/user/SocialMediaMarketingSkill/lab/edit-05/broll/wkc/logo.png")
    json.dump(cfg, open(f"{E}/out/edit_v22.json", "w"), indent=1)

    ev = [{"t": c["frame"] / FPS, "type": "shutter"} for c in caps + labels]
    ev += [{"t": round(cfg["end_card"]["start"] + 0.35, 4), "type": "shutter"}]
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{E}/out/v22_events.json", "w"), indent=1)

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
