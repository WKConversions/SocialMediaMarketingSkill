#!/usr/bin/env python3
"""Build a caption plan (plan.json) from word timings, following a style's plan_rules.

    python3 build_plan.py --words clip_words.json --style styles/prime.json --out plan.json [--video clip.mp4] \
        [--layout talking_head] [--lang nl] [--masks DIR] [--script w1,w2] [--behind w] [--fade w] [--glitch w1,w2] \
        [--entrance w=preset,...] [--no-sync]

Input words: [{"word": "testvideo", "start": 0.937, "end": 1.487}, ...] (faster-whisper style).
Output: {"style", "layout", "source", "words": [{"id", "text", "words", "start", "end", "class", "blend",
         "entrance", "behind_subject", "position", "zoom"}]}

Rules (all parameters come from style["plan_rules"]):
  0. Sync: with --video, word starts that whisper placed in a pause are moved to the first voiced onset
     (speech-band energy), so a caption never appears long before its word is said.
  1. Emphasis / script words: the last word always; then content words ranked by spoken duration x length,
     never function words, never adjacent, at least min_spacing_s apart, until ~target_ratio of the words.
     --script overrides the choice.
  2. Grouping: runs of function words are chunked right-to-left (pairs like 'at my', 'at a', 'in the');
     a caption still shorter than min_unit_s joins a function-word neighbour (max_unit_words); content words
     are never merged unless merge_content_words, and a lone short function word stands alone if it lasts
     min_unit_s_alone.
  3. Variants: pink difference for the first emphasis phrase, glitch entrance on the opening caption and on
     the caption after each script word, blur-in on the middle script words, punch-in zooms on script words
     with a settle-out after them; an opt-in behind-subject depth word (--behind, or enabled in the style).
"""
import argparse
import json
import os
import re
import subprocess

import numpy as np

STOPWORDS = {
    'nl': set('''de het een en of maar want dus dat die dit deze wat wie waar wanneer hoe om te van voor met
        naar op in uit aan bij tot door over onder als dan ook nog al wel niet geen zo er hier daar
        ik je jij u hij zij ze we wij jullie mij me mijn jouw zijn haar ons onze hun hen men
        is ben bent was waren wordt worden word heb hebt heeft hadden had kan kun kunt kon konden
        zal zult zullen zou zouden moet moeten mag mogen wil wilt gaat ga gaan doe doet even'''.split()),
    'en': set('''a an the and or but so that this these those what who where when how to of for with from in
        on at by as than then also too not no i i'm im me my you your he she it its we us our they them
        their is am are was were be been being have has had do does did can could will would shall should
        may might must just very really yet still even out up off why don't doesn't didn't can't won't isn't aren't
        wasn't weren't i've i'll i'd you're you've you'll you'd he's she's it's we're we've they're they've
        that's there's what's let's gonna wanna'''.split()),
}
NUMBER_WORDS = set('''zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty
    hundred thousand million billion nul een twee drie vier vijf zes zeven acht negen tien elf twaalf twintig
    dertig veertig vijftig honderd duizend miljoen miljard'''.split())


def clean(w):
    return re.sub(r"[^\w'’-]", '', w.lower()).replace('’', "'")


def guess_lang(words):
    toks = [clean(w['word']) for w in words]
    score = {k: sum(t in v for t in toks) for k, v in STOPWORDS.items()}
    return max(score, key=score.get)


def speech_energy(path, cfg):
    """Speech-band energy envelope of the audio in `path`: (dB per hop, floor dB, hop seconds)."""
    sr = 16000
    lo, hi = cfg.get('band_hz', [250, 3500])
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-vn', '-ac', '1', '-ar', str(sr),
           '-af', f'highpass=f={lo},lowpass=f={hi}', '-f', 's16le', '-']
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.int16).astype(np.float64) / 32768
    hop_s = cfg.get('hop_s', 0.01)
    hop = int(sr * hop_s)
    n = len(x) // hop
    db = 10 * np.log10((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)
    return db, float(np.percentile(db, cfg.get('floor_percentile', 10))), hop_s


def snap_onsets(words, video, cfg):
    """Move word starts that whisper put inside a pause to the first voiced onset (in place).
    Returns a list of (word, whisper_start, onset, new_start)."""
    db, floor, hop_s = speech_energy(video, cfg)
    voiced = db > floor + cfg.get('voiced_db_above_floor', 15)
    need = max(1, int(round(cfg.get('min_voiced_s', 0.03) / hop_s)))
    run = np.convolve(voiced.astype(int), np.ones(need, int), 'full')[need - 1:]   # run[i] = voiced[i:i+need].sum()
    ok = run >= need
    moved = []
    for n, w in enumerate(words):
        if n == 0 and cfg.get('skip_first', False):
            continue
        i0 = int(w['start'] / hop_s)
        i1 = min(int(w['end'] / hop_s), len(ok))
        if i0 >= i1 or ok[i0]:
            continue
        hits = np.nonzero(ok[i0:i1])[0]
        if not len(hits):
            continue
        onset = (i0 + hits[0]) * hop_s
        if onset - w['start'] > cfg.get('min_shift_s', 0.1):
            new = round(max(w['start'], onset - cfg.get('lead_s', 0.0)), 3)
            moved.append((w['word'], w['start'], round(onset, 3), new))
            w['start_whisper'] = w['start']
            w['start'] = new
    return moved


def pick_script(words, stop, rules, forced=None):
    n = len(words)
    if forced is not None:
        return {i for i, w in enumerate(words) if clean(w['word']) in forced}
    chosen = set()
    if rules.get('always_last', True) and n:
        chosen.add(n - 1)
    target = max(1, round(rules.get('target_ratio', 0.18) * n))
    cands = [i for i, w in enumerate(words)
             if clean(w['word']) not in stop and len(clean(w['word'])) >= rules.get('min_word_len', 4)]
    cands.sort(key=lambda i: -(words[i]['end'] - words[i]['start']) * len(clean(words[i]['word'])))
    for i in cands:
        if len(chosen) >= target:
            break
        if i in chosen:
            continue
        if rules.get('never_adjacent', True) and (i - 1 in chosen or i + 1 in chosen):
            continue
        if any(abs(words[i]['start'] - words[j]['start']) < rules.get('min_spacing_s', 0.9) for j in chosen):
            continue
        chosen.add(i)
    return chosen


def group(words, script, stop, rules):
    """Return a list of word-index lists (caption units)."""
    chunk = rules.get('function_word_chunk', 2)
    units, i, n = [], 0, len(words)
    while i < n:
        if i in script or clean(words[i]['word']) not in stop:
            units.append([i])
            i += 1
            continue
        j = i
        while j < n and j not in script and clean(words[j]['word']) in stop:
            j += 1
        run = list(range(i, j))
        parts = []
        while run:                                   # right-to-left: 'dit / is een'
            parts.insert(0, run[-chunk:])
            run = run[:-chunk]
        units.extend(parts)
        i = j
    # merge captions that would flash for less than min_unit_s
    min_s, max_w = rules.get('min_unit_s', 0.167), rules.get('max_unit_words', 3)
    merge_content = rules.get('merge_content_words', True)
    alone_s = rules.get('min_unit_s_alone', min_s)

    def dur(u):
        return words[u[-1]]['end'] - words[u[0]]['start']

    def func_only(u):
        return u[0] not in script and all(clean(words[i]['word']) in stop for i in u)

    def can_merge(u, nb, relaxed):
        if nb is None or nb[0] in script or len(nb) + len(u) > max_w:
            return False
        return relaxed or merge_content or (func_only(u) and func_only(nb))

    changed = True
    while changed:
        changed = False
        for k, u in enumerate(units):
            if dur(u) >= min_s or u[0] in script:
                continue
            prev = units[k - 1] if k > 0 else None
            nxt = units[k + 1] if k + 1 < len(units) else None
            ok_prev, ok_next = can_merge(u, prev, False), can_merge(u, nxt, False)
            if not (ok_prev or ok_next) and dur(u) < alone_s:          # too short to stand alone
                ok_prev, ok_next = can_merge(u, prev, True), can_merge(u, nxt, True)
            if ok_prev and (not ok_next or dur(prev) <= dur(nxt)):
                units[k - 1] = prev + u
            elif ok_next:
                units[k + 1] = u + nxt
            else:
                continue
            del units[k]
            changed = True
            break
    return units


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--words', required=True)
    ap.add_argument('--style', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--video', help='source video (recorded in the plan)')
    ap.add_argument('--layout', help='layout variant name to record in the plan')
    ap.add_argument('--lang', help='nl / en (default: guessed from the words)')
    ap.add_argument('--masks', help='subject mask folder; enables the behind_subject variant')
    ap.add_argument('--script', help='comma-separated words to force as script (overrides the auto pick)')
    ap.add_argument('--behind', help='comma-separated words to put behind the subject (overrides the rule)')
    ap.add_argument('--fade', help='comma-separated words that get the blur-in entrance (overrides the rule)')
    ap.add_argument('--glitch', help='comma-separated words that get the glitch entrance (overrides the rule)')
    ap.add_argument('--entrance', help='per-word entrance presets, e.g. "dit=glitch_rgb,editen=blur_in" (applied last)')
    ap.add_argument('--no-sync', action='store_true', help='keep whisper word starts as they are')
    args = ap.parse_args(argv)

    style = json.load(open(args.style))
    pr = style.get('plan_rules', {})
    words = json.load(open(args.words))
    sync = pr.get('sync', {})
    if args.video and sync.get('enabled', False) and not args.no_sync:
        for wd, a, on, b in snap_onsets(words, args.video, sync):
            print(f'sync: {wd!r} whisper start {a:.3f}s is in a pause; voiced onset {on:.3f}s -> caption starts {b:.3f}s')
    lang = args.lang or guess_lang(words)
    stop = STOPWORDS.get(lang, set())
    forced = set(clean(w) for w in args.script.split(',')) if args.script else None
    script = pick_script(words, stop, pr.get('emphasis', {}), forced)
    units = group(words, script, stop, pr.get('grouping', {}))
    var = pr.get('variants', {})
    emph = pr.get('emphasis', {})

    plan = []
    for k, u in enumerate(units):
        toks = [clean(words[i]['word']) for i in u]
        is_script = u[0] in script
        text = ' '.join(toks)
        if is_script and emph.get('capitalise', True) and not (emph.get('number_words_lowercase', True) and toks[0] in NUMBER_WORDS):
            text = text[:1].upper() + text[1:]
        plan.append(dict(id=f'u{k + 1:02d}', text=text, words=[words[i]['word'] for i in u],
                         start=round(words[u[0]]['start'], 3), end=round(words[u[-1]]['end'], 3),
                         **{'class': 'script' if is_script else 'sans'},
                         blend='difference', entrance='cut', behind_subject=False, position='default', zoom=None))
        if 'start_whisper' in words[u[0]]:
            plan[-1]['start_whisper'] = words[u[0]]['start_whisper']
    s_idx = [k for k, p in enumerate(plan) if p['class'] == 'script']

    # pink difference for the first emphasis phrase ('at my Prime')
    fe = var.get('first_emphasis_phrase')
    if fe and s_idx:
        k = s_idx[0]
        plan[k]['blend'] = fe['blend']
        for j in range(k - 1, max(-1, k - 1 - fe.get('include_preceding_function_units', 0)), -1):
            if all(clean(w) in stop for w in plan[j]['words']):
                plan[j]['blend'] = fe['blend']
            else:
                break
    # entrances
    names = lambda s: set(clean(w) for w in s.split(',')) if s else None
    fade_w, glitch_w = names(args.fade), names(args.glitch)
    if plan and var.get('opening_entrance'):
        plan[0]['entrance'] = var['opening_entrance']
    if var.get('after_script_entrance'):
        for k in s_idx:
            if k + 1 < len(plan) and plan[k + 1]['class'] == 'sans':
                plan[k + 1]['entrance'] = var['after_script_entrance']
    se = var.get('script_entrances', {})
    for n, k in enumerate(s_idx):
        role = 'first' if n == 0 else ('last' if n == len(s_idx) - 1 else 'middle')
        plan[k]['entrance'] = se.get(role, 'cut')
    if fade_w is not None or glitch_w is not None:
        for p in plan:
            key = clean(p['text'].split()[0])
            if fade_w is not None:
                p['entrance'] = 'blur_in' if key in fade_w else ('cut' if p['entrance'] == 'blur_in' else p['entrance'])
            if glitch_w is not None:
                p['entrance'] = 'glitch' if key in glitch_w else ('cut' if p['entrance'] == 'glitch' else p['entrance'])
    # behind-subject depth word
    bs = var.get('behind_subject')
    behind_w = names(args.behind)
    has_masks = bool(args.masks and os.path.isdir(args.masks) and os.listdir(args.masks))
    if has_masks and ((bs and bs.get('enabled', True)) or behind_w):
        if behind_w is not None:
            picks = [k for k, p in enumerate(plan) if clean(p['text'].split()[0]) in behind_w]
        else:
            picks = s_idx[:bs.get('max_units', 1)] if bs.get('pick') == 'first_script' else []
        for k in picks:
            plan[k]['behind_subject'] = True
            plan[k]['position'] = (bs or {}).get('position', 'default')
    # explicit per-word entrance presets win over the rules
    if args.entrance:
        want = dict(kv.split('=', 1) for kv in args.entrance.split(','))
        want = {clean(k): v for k, v in want.items()}
        unknown = set(want.values()) - set(style.get('entrances', {}))
        if unknown:
            raise SystemExit(f'unknown entrance preset(s): {sorted(unknown)}')
        for p in plan:
            key = clean(p['text'].split()[0])
            if key in want:
                p['entrance'] = want[key]
    # digital zoom events
    zv = var.get('zoom', {})
    if zv:
        for n, k in enumerate(s_idx):
            plan[k]['zoom'] = zv.get('script_last') if n == len(s_idx) - 1 else zv.get('script')
            if plan[k]['zoom'] == 'none':
                plan[k]['zoom'] = None
        if plan and plan[0]['zoom'] is None and zv.get('first_unit'):
            plan[0]['zoom'] = zv['first_unit']
        for k, p in enumerate(plan[:-1]):
            if p['zoom'] in ('punch_in', 'opening_punch') and plan[k + 1]['zoom'] is None and zv.get('after_punch'):
                plan[k + 1]['zoom'] = zv['after_punch']

    out = dict(style=style.get('name'), layout=args.layout, language=lang,
               source=dict(video=args.video, words=args.words),
               rules='built by tools/build_plan.py from style.plan_rules',
               words=plan)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    n_script = sum(p['class'] == 'script' for p in plan)
    print(f'{len(words)} words -> {len(plan)} captions, {n_script} script ({lang}):')
    for p in plan:
        print(f"  {p['start']:6.3f}-{p['end']:6.3f}  {p['text']:<12s} {p['class']:6s} {p['blend']:16s} {p['entrance']:8s}"
              f" behind={p['behind_subject']!s:5s} pos={p['position']:<11s} zoom={p['zoom']}")


if __name__ == '__main__':
    main()
