"""The caption plan: which words form each caption, its class and colour, when it appears, and its sounds.

plan.json is the single interface between the steps. The renderer, the effects track and the verification
all read it, so captions and sounds cannot drift apart. It is also the place to make edits by hand (run
edit_video.py with --plan-only, edit, then run it again with --plan):
  * wording:   change a caption's "text" (spoken "words" stay as they are, they are the timing source)
  * emphasis:  set "class" to "script" or "sans" ("fill" "pink"/"white"); a script caption gets a hit sound
  * grouping:  move word objects between captions; delete "start" of a changed caption to re-derive it
  * timing:    "start" (seconds in the clip); a caption shows until the next one starts
Rules enforced on every load (normalise): every caption gets the shutter; script captions get a hit, sans
captions never do; captions must start on different frames; text case follows the class.

Default choices (all parameters from the style):
  emphasis  about 1 in emphasis.every_n_words words; content words (not function words) with at least
            min_letters letters, ranked by letters x spoken duration; the video's last word first if it
            qualifies; never two in a row. First emphasis word pink, the others white.
  grouping  emphasis and content words stand alone; each run of function words between them is one caption
            (split evenly above max_words); a caption on screen for fewer than min_frames frames merges
            into a neighbouring non-emphasis caption (function-word neighbours first) within max_words.
  timing    caption start = first word start - caption_lead_frames; it holds until the next caption.
  sound     per caption a slightly different shutter pitch/level/pan (seeded, so re-runs are identical).
"""
import math
import re

import numpy as np

from common import PipelineError, first_frame

EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
GUIDE = [
    "Edit this file, then run: edit_video.py INPUT --style STYLE --out DIR --plan THIS_FILE",
    "wording: change a caption's text; its words stay as spoken (they are the timing source)",
    "emphasis: class script or sans, fill pink or white; script captions get the punch-in zoom and a hit sound",
    "grouping: move word objects between captions; remove start from a changed caption to re-derive it",
    "timing: start is seconds in the clip; a caption shows until the next caption starts",
    "always enforced: every caption gets the shutter, script captions a hit, sans captions none; captions start "
    "on different frames; text case follows the class",
]


def display_word(raw):
    """A transcript word as it appears in a caption: no surrounding punctuation, inner ' and - kept."""
    return EDGE_PUNCT.sub("", raw.replace("’", "'"))


def key(raw):
    return display_word(raw).lower()


def letters(raw):
    return sum(c.isalpha() for c in raw)


def apply_case(text, case):
    text = text.lower()
    return text[:1].upper() + text[1:] if case == "capitalise" else text


# ------------------------------------------------------------------------------------------- emphasis
def pick_emphasis(words, function_words, cfg, forced=None):
    """Indices of the emphasis (script) words."""
    if forced is not None:
        chosen = {i for i, w in enumerate(words) if key(w["word"]) in forced}
        missing = forced - {key(w["word"]) for w in words}
        if missing:
            raise PipelineError(f"--emphasis word(s) not in the transcript: {', '.join(sorted(missing))}")
        return chosen
    n = len(words)
    target = max(1, round(n / cfg["every_n_words"]))
    cands = [i for i, w in enumerate(words)
             if key(w["word"]) not in function_words and letters(w["word"]) >= cfg["min_letters"]]
    order = sorted(cands, key=lambda i: -letters(words[i]["word"]) * (words[i]["end"] - words[i]["start"]))
    if cfg.get("prefer_last_word") and n - 1 in cands:
        order = [n - 1] + [i for i in order if i != n - 1]
    chosen = set()
    for i in order:
        if len(chosen) >= target:
            break
        if cfg.get("never_adjacent") and (i - 1 in chosen or i + 1 in chosen):
            continue
        chosen.add(i)
    return chosen


# ------------------------------------------------------------------------------------------- grouping
def caption_start(word_start, lead_frames, fps):
    return round(max(0.0, word_start - lead_frames / fps), 3)


def group(words, emphasis, function_words, cfg, lead_frames, fps, n_frames):
    """Word-index lists, one per caption."""
    units, i = [], 0
    while i < len(words):
        if i in emphasis or key(words[i]["word"]) not in function_words:
            units.append([i])
            i += 1
            continue
        j = i
        while j < len(words) and j not in emphasis and key(words[j]["word"]) in function_words:
            j += 1
        run = list(range(i, j))                                  # a run of function words
        parts = math.ceil(len(run) / cfg["max_words"])
        sizes = [len(run) // parts + (k < len(run) % parts) for k in range(parts)]
        for size in sizes:
            units.append(run[:size])
            run = run[size:]
        i = j

    def frames_shown():
        ff = [first_frame(caption_start(words[u[0]]["start"], lead_frames, fps), fps) for u in units] + [n_frames]
        return [ff[k + 1] - ff[k] for k in range(len(units))]

    def function_only(u):
        return all(key(words[i]["word"]) in function_words for i in u)

    while True:
        shown = frames_shown()
        merged = False
        for k, u in enumerate(units):
            if shown[k] >= cfg["min_frames"] or u[0] in emphasis:
                continue
            nbs = [nb for nb in (k - 1, k + 1) if 0 <= nb < len(units) and units[nb][0] not in emphasis
                   and (len(units[nb]) + len(u) <= cfg["max_words"] or shown[k] == 0)]
            if not nbs:
                continue
            nb = min(nbs, key=lambda m: (not function_only(units[m]), shown[m]))
            lo, hi = min(k, nb), max(k, nb)
            units[lo:hi + 1] = [units[lo] + units[hi]]
            merged = True
            break
        if not merged:
            return units


# ------------------------------------------------------------------------------------------- build
def shutter_variation(rng, cfg):
    """Slightly different pitch, level and pan for each use of the shutter (drawn in the lab's order)."""
    pan = rng.uniform(*cfg["pan_jitter"])
    gain = cfg["gain_db"] + rng.uniform(*cfg["gain_jitter_db"])
    pitch = rng.uniform(*cfg["pitch_jitter"])
    return dict(pitch=round(float(pitch), 4), gain_db=round(float(gain), 2), pan=round(float(pan), 3))


def hit_for(role, cfg):
    r = cfg["roles"][role]
    return dict(role=role, size=r["size"], tail=r["tail"],
                gain_db=round(cfg["gain_db"] + cfg["gain_db_per_size"] * (r["size"] - 1), 2))


def build(words, language, transcript, style, style_name, clip, emphasis=None):
    """words: aligned words in clip time [{"word", "start", "end", "score"}] -> plan dict."""
    function_words = set(style["grouping"]["function_words"].get(language, []))
    if not function_words:
        raise PipelineError(f"style has no function-word list for language {language!r}")
    words = [dict(w, word=display_word(w["word"])) for w in words if display_word(w["word"])]
    if not words:
        raise PipelineError("no words to caption")
    lead = style["timing"]["caption_lead_frames"]
    emph = pick_emphasis(words, function_words, style["emphasis"], emphasis)
    units = group(words, emph, function_words, style["grouping"], lead, clip.fps, clip.n_frames)
    captions = []
    for u in units:
        cls = "script" if u[0] in emph else "sans"
        captions.append(dict(text=apply_case(" ".join(words[i]["word"] for i in u), style["classes"][cls]["case"]),
                             words=[words[i] for i in u],
                             start=caption_start(words[u[0]]["start"], lead, clip.fps),
                             **{"class": cls}))
    plan = dict(about=GUIDE, style=style_name, language=language, transcript=transcript,
                clip=clip.to_dict(), captions=captions)
    rng = np.random.default_rng(style["sfx"]["seed"])
    for c in captions:
        c["sfx"] = dict(shutter=shutter_variation(rng, style["sfx"]["shutter"]))
    normalise(plan, style)
    return plan


# ------------------------------------------------------------------------------------------- rules
def normalise(plan, style):
    """Fill what a hand-edited plan leaves out and enforce the style's rules (in place).
    Returns a list of notes about what was changed."""
    caps = plan.get("captions") or []
    if not caps:
        raise PipelineError("the plan has no captions")
    fps, n_frames = plan["clip"]["fps"], plan["clip"]["frames"]
    lead = style["timing"]["caption_lead_frames"]
    sfx = style["sfx"]
    notes = []
    script_idx = [k for k, c in enumerate(caps) if c.get("class", "sans") == "script"]
    for k, c in enumerate(caps):
        if not c.get("words"):
            raise PipelineError(f"caption {k + 1} ({c.get('text')!r}) has no words")
        c.setdefault("class", "sans")
        if c["class"] not in style["classes"]:
            raise PipelineError(f"caption {k + 1}: unknown class {c['class']!r}")
        c.setdefault("text", " ".join(w["word"] for w in c["words"]))
        cased = apply_case(c["text"], style["classes"][c["class"]]["case"])
        if cased != c["text"]:
            notes.append(f"caption {k + 1}: text case follows its class ({c['text']!r} -> {cased!r})")
            c["text"] = cased
        if c.get("start") is None:
            c["start"] = caption_start(c["words"][0]["start"], lead, fps)
        if "fill" not in c:
            fills = style["emphasis"]["fills"]
            c["fill"] = (fills["first"] if script_idx and k == script_idx[0] else fills["others"]) \
                if c["class"] == "script" else style["emphasis"]["sans_fill"]
        if c["fill"] not in style["colours"]:
            raise PipelineError(f"caption {k + 1}: unknown fill {c['fill']!r} (style colours: {', '.join(style['colours'])})")
        # every caption gets the shutter; script captions add a hit, sans captions never do
        c.setdefault("sfx", {})
        if not c["sfx"].get("shutter"):
            c["sfx"]["shutter"] = shutter_variation(np.random.default_rng([sfx["seed"], k]), sfx["shutter"])
            notes.append(f"caption {k + 1}: shutter added (every caption gets one)")
        if c["class"] == sfx["hit"]["on"]:
            if not c["sfx"].get("hit"):
                n = script_idx.index(k)
                role = "first" if n == 0 else ("last" if n == len(script_idx) - 1 else "middle")
                c["sfx"]["hit"] = hit_for(role, sfx["hit"])
        elif c["sfx"].get("hit"):
            notes.append(f"caption {k + 1}: hit removed (only {sfx['hit']['on']} captions get one)")
            c["sfx"]["hit"] = None
        c["sfx"].setdefault("hit", None)
    frames = [first_frame(c["start"], fps) for c in caps]
    for k in range(len(caps) - 1):
        if frames[k + 1] <= frames[k]:
            raise PipelineError(f"captions {k + 1} ({caps[k]['text']!r}) and {k + 2} ({caps[k + 1]['text']!r}) would "
                                f"start on frames {frames[k]} and {frames[k + 1]}; merge them or move a start")
    if frames[-1] >= n_frames:
        raise PipelineError(f"the last caption starts after the clip ends (frame {frames[-1]} of {n_frames})")
    for k, c in enumerate(caps):
        c["end"] = caps[k + 1]["start"] if k + 1 < len(caps) else round(n_frames / fps, 3)
    # keep the keys in a readable order
    order = ["text", "class", "fill", "start", "end", "sfx", "words"]
    plan["captions"] = [{kk: c[kk] for kk in order + [x for x in c if x not in order]} for c in caps]
    return notes


def spoken_words(plan):
    """All spoken words in order, and the index of each caption's first word."""
    words, firsts = [], []
    for c in plan["captions"]:
        firsts.append(len(words))
        words += c["words"]
    return words, firsts


def describe(plan):
    fps = plan["clip"]["fps"]
    lines = [f"{len(plan['captions'])} captions, {plan['clip']['frames']} frames at {fps} fps ({plan['language']}):"]
    for c in plan["captions"]:
        hit = c["sfx"]["hit"]
        lines.append(f"  frame {first_frame(c['start'], fps):4d}  {c['start']:6.3f}s  {c['text']:<14s} {c['class']:6s} "
                     f"{c['fill']:5s} shutter{'+hit(' + hit['role'] + ')' if hit else ''}")
    return "\n".join(lines)
