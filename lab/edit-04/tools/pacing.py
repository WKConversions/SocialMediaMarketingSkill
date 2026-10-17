"""Consistent pacing for the Brooklyn Bridge / WKC edit.

Every pause longer than MAX_GAP (inside a clip and between clips) becomes the same short gap: keep a
short tail after the last word and a short lead-in before the next word. Cut points are frame-accurate
so picture and sound stay locked. The client asked to drop "sales is marketing" in 3333.
Writes out/edl.json (kept segments, words and their output times) and out/voice.wav.
"""
import json
import math
import subprocess

E = "/home/user/SocialMediaMarketingSkill/lab/edit-04"
FPS = 30
CLIPS = ["1111", "2222", "3333", "4444"]
MAX_GAP = 0.22          # pauses longer than this get normalised
TAIL, LEAD = 0.09, 0.07  # kept after a word end / before a word start  ->  ~0.16-0.2 s pause, everywhere
DROP = {"3333": [(9, 11)]}   # word index ranges to remove ("sales is marketing")
DURATION = {"1111": 4.8333, "2222": 16.4667, "3333": 8.9333, "4444": 2.7667}


def fl(t):
    return math.floor(t * FPS + 1e-6) / FPS


def ce(t):
    return math.ceil(t * FPS - 1e-6) / FPS


def main():
    segments, words_out = [], []
    out_t = 0.0
    prev_clip_tail = None
    for ci, clip in enumerate(CLIPS):
        words = json.load(open(f"{E}/footage/{clip}.aligned.json"))
        drop = {i for a, b in DROP.get(clip, []) for i in range(a, b + 1)}
        kept = [dict(w, idx=i) for i, w in enumerate(words) if i not in drop]
        # split this clip's kept words into runs separated by long pauses (or by dropped words)
        runs, cur = [], [kept[0]]
        for a, b in zip(kept, kept[1:]):
            if b["start"] - a["end"] > MAX_GAP or b["idx"] != a["idx"] + 1 and b["start"] - a["end"] > 0.12:
                runs.append(cur)
                cur = [b]
            else:
                cur.append(b)
        runs.append(cur)
        for ri, run in enumerate(runs):
            first_clip_run = ci == 0 and ri == 0
            last_clip_run = ci == len(CLIPS) - 1 and ri == len(runs) - 1
            s = fl(max(0.0, run[0]["start"] - (0.08 if first_clip_run else LEAD)))
            e = ce(min(DURATION[clip], run[-1]["end"] + (0.30 if last_clip_run else TAIL)))
            seg = dict(clip=clip, src_start=round(s, 4), src_end=round(e, 4), out_start=round(out_t, 4), out_end=round(out_t + e - s, 4))
            segments.append(seg)
            for w in run:
                words_out.append(dict(word=w["word"], clip=clip, idx=w["idx"], score=w["score"],
                                      src_start=w["start"], src_end=w["end"],
                                      start=round(out_t + w["start"] - s, 4), end=round(out_t + w["end"] - s, 4)))
            out_t += e - s
    # pauses actually left between runs (for the report)
    gaps = []
    for a, b in zip(segments, segments[1:]):
        wa = [w for w in words_out if w["start"] < a["out_end"] and w["start"] >= a["out_start"]][-1]
        wb = [w for w in words_out if w["start"] >= b["out_start"]][0]
        gaps.append(round(wb["start"] - wa["end"], 3))
    json.dump(dict(fps=FPS, segments=segments, words=words_out, duration=round(out_t, 4), gaps=gaps), open(f"{E}/out/edl.json", "w"), indent=1)
    # voice track: each segment from its clip, 8 ms fades at every cut, concatenated
    inputs, chains = [], []
    for k, seg in enumerate(segments):
        inputs += ["-i", f"{E}/footage/{seg['clip']}.mov"]
        d = seg["src_end"] - seg["src_start"]
        chains.append(f"[{k}:a]atrim=start={seg['src_start']}:end={seg['src_end']},asetpts=PTS-STARTPTS,aresample=48000,"
                      f"afade=t=in:d=0.008,afade=t=out:st={d - 0.008:.4f}:d=0.008[a{k}]")
    graph = ";".join(chains) + ";" + "".join(f"[a{k}]" for k in range(len(segments))) + f"concat=n={len(segments)}:v=0:a=1[out]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]", "-ac", "1", "-c:a", "pcm_s16le",
                    f"{E}/out/voice.wav"], check=True)
    print(f"{len(segments)} segments, {out_t:.3f} s (was {sum(DURATION.values()):.2f} s)")
    print("pauses between kept runs (s):", gaps)
    for s in segments:
        print(f"  {s['clip']} {s['src_start']:6.3f}-{s['src_end']:6.3f}  ->  {s['out_start']:6.3f}-{s['out_end']:6.3f}")


if __name__ == "__main__":
    main()
