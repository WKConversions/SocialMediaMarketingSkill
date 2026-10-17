"""The caption plan: which words form each caption, its class and colour, when it appears, and its sounds.

plan.json is the single interface between the steps. The renderer, the effects track and the verification
all read it, so captions and sounds cannot drift apart. It is also the place to make edits by hand (run
edit_video.py with --plan-only, edit, then run it again with --plan):
  * wording:   change a caption's "text" (spoken "words" stay as they are, they are the timing source)
  * emphasis:  set "class" to "script" or "sans"; a script caption gets the punch-in zoom and a hit sound
  * colour:    follows the class (first script caption pink, other script and all sans captions white);
               to force another colour set "fill" and "fill_override": true on that caption
  * grouping:  move word objects between captions; delete "start" of a changed caption to re-derive it
  * timing:    "start" (seconds in the clip); a caption shows until the next one starts
Rules enforced on every load (normalise): every caption gets the shutter; script captions get a hit, sans
captions never do; text case and colour follow the class; every caption is on screen for at least
grouping.min_frames frames (a plan that breaks this is written out but not rendered: edit it).

Default choices (all parameters from the style):
  emphasis  about 1 in emphasis.every_n_words words; content words (not function words) with at least
            min_letters letters, ranked by letters x spoken duration; the video's last word first if it
            qualifies; never two in a row, and never so close that the caption(s) between them, before the
            first or after the last, or the emphasis caption itself would be on screen for fewer than
            grouping.min_frames frames (emphasis captions stand alone, so nothing could merge them away);
            an emphasis word next to a caption that grouping still cannot bring to min_frames is passed over
            for the next candidate. First emphasis word pink, the others white.
  grouping  emphasis and content words stand alone; each run of function words between them is one caption
            (split evenly above max_words); a caption on screen for fewer than min_frames frames merges
            into a neighbouring non-emphasis caption (function-word neighbours first) within max_words, or,
            when both neighbours are full, takes over the nearest word of one of them if both then last.
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
    "emphasis: class script or sans; script captions get the punch-in zoom and a hit sound",
    "colour: follows the class (first script caption pink, the rest white); to force another colour on a caption "
    "set fill and fill_override: true",
    "grouping: move word objects between captions; remove start from a changed caption to re-derive it",
    "timing: start is seconds in the clip; a caption shows until the next caption starts",
    "always enforced: every caption gets the shutter, script captions a hit, sans captions none; text case and "
    "colour follow the class; every caption is on screen for at least the style's grouping.min_frames frames",
    "quality: how sure the transcript and its timing were when this plan was made; if that stopped the run, check "
    "the words and their times (or run again with --lang), then render with --plan (it is not checked again)",
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
def word_frames(words, lead_frames, fps, n_frames):
    """The frame on which a caption starting with each word would appear, plus n_frames at the end."""
    return [first_frame(caption_start(w["start"], lead_frames, fps), fps) for w in words] + [n_frames]


def pick_emphasis(words, function_words, cfg, min_frames, frames, forced=None, passed_over=()):
    """Indices of the emphasis (script) words. frames: word_frames(...); passed_over: word indices not to use."""
    if forced is not None:
        forced = {key(f) for f in forced}
        chosen = {i for i, w in enumerate(words) if key(w["word"]) in forced}
        missing = forced - {key(w["word"]) for w in words}
        if missing:
            raise PipelineError(f"--emphasis word(s) not in the transcript: {', '.join(sorted(missing))}")
        return chosen
    n = len(words)
    target = max(1, round(n / cfg["every_n_words"]))
    cands = [i for i, w in enumerate(words) if i not in passed_over
             and key(w["word"]) not in function_words and letters(w["word"]) >= cfg["min_letters"]]
    order = sorted(cands, key=lambda i: -letters(words[i]["word"]) * (words[i]["end"] - words[i]["start"]))
    if cfg.get("prefer_last_word") and n - 1 in cands:
        order = [n - 1] + [i for i in order if i != n - 1]

    def fits(i, chosen):
        """An emphasis caption stands alone and nothing merges into it, so the word itself and the stretch of
        other words on each side (up to the neighbouring emphasis word or the clip's edge) must each stay on
        screen for min_frames frames; an empty stretch between two emphasis words means two in a row."""
        left = max((j for j in chosen if j < i), default=-1)
        right = min((j for j in chosen if j > i), default=n)
        if frames[i + 1] - frames[i] < min_frames:
            return False
        for a, b, bounded in ((left + 1, i, left >= 0), (i + 1, right, right < n)):
            if a == b and bounded and cfg.get("never_adjacent"):
                return False
            if a < b and frames[b] - frames[a] < min_frames:
                return False
        return True

    chosen = set()
    for i in order:
        if len(chosen) >= target:
            break
        if fits(i, chosen):
            chosen.add(i)
    return chosen


# ------------------------------------------------------------------------------------------- grouping
def caption_start(word_start, lead_frames, fps):
    return round(max(0.0, word_start - lead_frames / fps), 3)


def group(words, emphasis, function_words, cfg, frames):
    """Word-index lists, one per caption. frames: word_frames(...)."""
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
        ff = [frames[u[0]] for u in units] + [frames[-1]]
        return [ff[k + 1] - ff[k] for k in range(len(units))]

    def function_only(u):
        return all(key(words[i]["word"]) in function_words for i in u)

    def take_word(k):
        """Short unit k between full neighbours: take over the nearest word of a non-emphasis neighbour if both
        units then stay on screen for min_frames frames."""
        for nb in (k - 1, k + 1):
            if not 0 <= nb < len(units) or units[nb][0] in emphasis or len(units[nb]) < 2:
                continue
            trial = units[:]
            if nb < k:
                trial[nb], trial[k] = units[nb][:-1], units[nb][-1:] + units[k]
            else:
                trial[k], trial[nb] = units[k] + units[nb][:1], units[nb][1:]
            ff = [frames[u[0]] for u in trial] + [frames[-1]]
            if len(trial[k]) <= cfg["max_words"] and min(ff[j + 1] - ff[j] for j in (k, nb)) >= cfg["min_frames"]:
                units[:] = trial
                return True
        return False

    while True:
        shown = frames_shown()
        merged = False
        for k, u in enumerate(units):
            if shown[k] >= cfg["min_frames"] or u[0] in emphasis:
                continue
            nbs = [nb for nb in (k - 1, k + 1) if 0 <= nb < len(units) and units[nb][0] not in emphasis
                   and (len(units[nb]) + len(u) <= cfg["max_words"] or shown[k] == 0)]
            if not nbs:
                if take_word(k):
                    merged = True
                    break
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
    """Slightly different pitch, level and pan for each use of the shutter, within the style's ranges."""
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
    frames = word_frames(words, lead, clip.fps, clip.n_frames)
    min_f = style["grouping"]["min_frames"]
    passed_over = set()
    while True:
        emph = pick_emphasis(words, function_words, style["emphasis"], min_f, frames, emphasis, passed_over)
        units = group(words, emph, function_words, style["grouping"], frames)
        # an emphasis caption next to a caption that grouping could not bring to min_frames is passed over for
        # another word (a forced --emphasis is kept: the plan then lists the problem)
        ff = [frames[u[0]] for u in units] + [frames[-1]]
        stuck = {units[nb][0] for k in range(len(units)) if ff[k + 1] - ff[k] < min_f
                 for nb in (k - 1, k + 1) if 0 <= nb < len(units) and units[nb][0] in emph}
        if emphasis is not None or not stuck:
            break
        passed_over |= stuck
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
    Returns a list of notes about what was changed. Rules that cannot be fixed automatically (a caption on
    screen for fewer than grouping.min_frames frames) are listed in plan["problems"]: the plan can still be
    written out and edited, but the CLI does not render it."""
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
        # colour follows the class on every load (first script caption, other script captions, sans), so it
        # cannot go stale when emphasis is moved by hand; fill_override keeps a deliberate other colour
        fills = style["emphasis"]["fills"]
        derived = (fills["first"] if k == script_idx[0] else fills["others"]) if c["class"] == "script" \
            else style["emphasis"]["sans_fill"]
        if c.get("fill_override"):
            if c.get("fill") not in style["colours"]:
                raise PipelineError(f"caption {k + 1}: fill_override needs a fill from the style's colours "
                                    f"({', '.join(style['colours'])}), not {c.get('fill')!r}")
            if c["fill"] != derived:
                notes.append(f"caption {k + 1}: fill {c['fill']!r} kept by fill_override (the style would use {derived!r})")
        else:
            if c.get("fill") not in (None, derived):
                notes.append(f"caption {k + 1}: fill follows its class ({c['fill']!r} -> {derived!r}); "
                             "set fill_override to keep another colour")
            c["fill"] = derived
            c.pop("fill_override", None)
        # every caption gets the shutter; script captions add a hit, sans captions never do
        c.setdefault("sfx", {})
        if not c["sfx"].get("shutter"):
            c["sfx"]["shutter"] = shutter_variation(np.random.default_rng([sfx["seed"], k]), sfx["shutter"])
            notes.append(f"caption {k + 1}: shutter added (every caption gets one)")
        if c["class"] == sfx["hit"]["on"]:
            # the hit's role (size, tail) follows the order of the script captions, like the colour
            n = script_idx.index(k)
            role = "first" if n == 0 else ("last" if n == len(script_idx) - 1 else "middle")
            old = c["sfx"].get("hit")
            if not old or old.get("role") != role:
                if old:
                    notes.append(f"caption {k + 1}: hit follows its place among the script captions "
                                 f"({old.get('role')!r} -> {role!r})")
                c["sfx"]["hit"] = hit_for(role, sfx["hit"])
        elif c["sfx"].get("hit"):
            notes.append(f"caption {k + 1}: hit removed (only {sfx['hit']['on']} captions get one)")
            c["sfx"]["hit"] = None
        c["sfx"].setdefault("hit", None)
    # every caption on screen for at least min_frames frames (and so the captions start on different frames)
    frames = [first_frame(c["start"], fps) for c in caps] + [n_frames]
    min_f = style["grouping"]["min_frames"]
    problems = []
    for k, c in enumerate(caps):
        c["end"] = caps[k + 1]["start"] if k + 1 < len(caps) else round(n_frames / fps, 3)
        c["frames_on_screen"] = shown = frames[k + 1] - frames[k]
        if shown < min_f:
            nb = " or ".join(f"{j + 1} ({caps[j]['text']!r}, {caps[j]['class']})" for j in (k - 1, k + 1)
                             if 0 <= j < len(caps))
            problems.append(f"caption {k + 1} ({c['text']!r}, frame {frames[k]}) is on screen for {shown} frame(s), the "
                            f"style's minimum is {min_f}: move its words into caption {nb} (make a script caption "
                            f"that takes them sans), or move a start")
    if problems:
        plan["problems"] = problems
    else:
        plan.pop("problems", None)
    # keep the keys in a readable order
    order = ["text", "class", "fill", "start", "end", "frames_on_screen", "sfx", "words"]
    plan["captions"] = [{kk: c[kk] for kk in order + [x for x in c if x not in order]} for c in caps]
    return notes


def pauses_not_cut(words, cfg):
    """PREFERENCES.md 'Editing' asks to cut pauses longer than ~0.3 s (with a view switch, negative frames and a
    flash). Prime is packaged as the approved v11, which keeps them, so until the client decides they are only
    listed: every gap longer than cfg["pause_s"] between consecutive spoken words, with cfg["pause_status"]."""
    return [dict(after=a["word"], before=b["word"], start=a["end"], end=b["start"], length_s=round(b["start"] - a["end"], 3),
                 status=cfg["pause_status"]) for a, b in zip(words, words[1:]) if b["start"] - a["end"] > cfg["pause_s"]]


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
        lines.append(f"  frame {first_frame(c['start'], fps):4d}  {c['start']:6.3f}s  {c['frames_on_screen']:3d} fr  "
                     f"{c['text']:<14s} {c['class']:6s} {c['fill']:5s} shutter{'+hit(' + hit['role'] + ')' if hit else ''}")
    return "\n".join(lines)
