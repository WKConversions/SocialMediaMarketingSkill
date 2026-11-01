"""Creative decisions for v1 of the Wright brothers 1903 -> WKC edit (30 fps), the new classy / jazz style.

The client's brief: Sinatra "Come Fly With Me" from 0.5 s before "Come fly with me, let's fly, let's fly away";
interludes between his explaining where the music rises over real images and film of what he explains, then a smooth
cross-dissolve back to him; ~50% explaining, ~30% music + visuals, ~20% visuals under his voice, spread out; no
flashes, no motion design; one classy serif caption style in one consistent spot; a softer shutter on every caption.

The paced voice (out/edl.json, 26.0 s) is split into four blocks at its sentence pauses and laid on the video
timeline with interludes in between. All times here are VIDEO seconds unless named voice time. Visual slots are filled
from out/media.json (fact-checked, commercially cleared items only). Writes out/edit_v19.json + out/v19_events.json.
"""
import json
import os

E = "/home/user/SocialMediaMarketingSkill/lab/edit-09"
FPS = 30
SONG_START_S = 9.59                     # music/library.json "sinatra-come-fly-with-me" start_offset_s

# (voice start, voice end, video start frame): block boundaries sit in the pacing pauses between sentences
BLOCKS = [(0.0, 5.6, 171),              # "In 1903 two brothers ... a machine that could fly"   (after "...fly away")
          (5.6, 13.5333, 393),          # "The first flight ... couldn't have been explained earlier"
          (13.5333, 17.3333, 697),      # "A machine lifted off the ground. And that's the power of visualization"
          (17.3333, 26.0, 865)]         # "So the next time you take a flight ... instead of explaining"
END_CARD_S = 2.0

# interludes (music up, visuals): start/end are filled from the blocks; slot sequence with relative weights
INTERLUDES = [
    dict(name="I0", slots=[("glider", 0.55), ("brothers", 0.45)]),   # Kitty Hawk; Wilbur and Orville
    dict(name="I1", slots=[("flyer_camp", 1.0)]),                     # "...a machine that could fly" -> the machine
    dict(name="I2", slots=[("photo", 1.0)]),                          # THE photograph, on Sinatra's "Come fly with me"
    dict(name="I3", slots=[("film", 1.0)]),                           # the machine flying (real film, later years)
]
# visuals under his voice (voice time); 'carry' = continues the interlude visual before/after it
UNDER_VOICE = [
    (11.94, 13.5333, "patent"),         # "have been explained earlier": the patent drawing (explaining on paper), into I2
    (13.5333, 15.6333, "photo"),        # "A machine lifted off the ground": the photograph stays (carry from I2)
    (17.3333, 19.0667, "film"),         # "So the next time you take a flight": the film continues (carry from I3)
    (23.5667, 24.80, "photo_callback"),  # "visualized the product": the photograph once more
]
# speaker framing per paced segment (index into edl["segments"]): every jump cut changes the framing
VIEWS = ["wide", "medium", "wide", "medium", "wide", "medium", "medium", "wide", "medium", "wide", "medium"]

CAPTIONS = [  # (text, index of its first word in edl["words"])
    ("In 1903", 0), ("two brothers achieved", 4), ("something extraordinary", 7), ("They built a machine", 9),
    ("that could fly", 13),
    ("The first flight", 16), ("took only 12 seconds", 19), ("But during those 12 seconds", 23),
    ("one photograph explained", 28), ("something that couldn't", 31), ("have been explained earlier", 34),
    ("A machine", 38), ("lifted off the ground", 40), ("And that's the power", 44), ("of visualization", 48),
    ("So the next time", 50), ("you take a flight", 54), ("note that the reason", 58), ("you can take that flight", 62),
    ("is because someone", 67), ("at some point in time", 70), ("visualized the product", 75), ("instead of explaining", 78),
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
        assert abs(b["end"] * FPS - round(b["end"] * FPS)) < 1e-3, b
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
            labels.append(dict(text=sh["label"], frame=f0, end_frame=int(round(sh["end"] * FPS)) - 4, fade_out=True, label=True))

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
               end_card=dict(start=blocks[-1]["end"], text=["Some things are better", "shown than explained"]),
               logo="/home/user/SocialMediaMarketingSkill/lab/edit-05/broll/wkc/logo.png")
    json.dump(cfg, open(f"{E}/out/edit_v19.json", "w"), indent=1)

    ev = [{"t": c["frame"] / FPS, "type": "shutter"} for c in caps + labels]
    ev += [{"t": round(cfg["end_card"]["start"] + 0.35, 4), "type": "shutter"}]
    json.dump(sorted(ev, key=lambda e: e["t"]), open(f"{E}/out/v19_events.json", "w"), indent=1)

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
