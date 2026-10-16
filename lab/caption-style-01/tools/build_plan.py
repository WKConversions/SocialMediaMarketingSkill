#!/usr/bin/env python3
"""Build a caption plan (plan.json) from word timings, following a style's plan_rules.

    python3 build_plan.py --words clip_words.json --style styles/prime.json --out plan.json \
        [--layout talking_head] [--lang nl] [--masks DIR] [--script w1,w2] [--behind w] [--fade w] [--glitch w1,w2]

Input words: [{"word": "testvideo", "start": 0.937, "end": 1.487}, ...] (faster-whisper style).
Output: {"style", "layout", "source", "words": [{"id", "text", "words", "start", "end", "class", "blend",
         "entrance", "behind_subject", "position", "zoom"}]}

Rules (all parameters come from style["plan_rules"]):
  1. Emphasis / script words: the last word always; then content words ranked by spoken duration x length,
     never function words, never adjacent, at least min_spacing_s apart, until ~target_ratio of the words.
     --script overrides the choice.
  2. Grouping: runs of function words are chunked right-to-left (pairs like 'at my', 'at a', 'in the');
     any caption still shorter than min_unit_s joins a non-script neighbour (max_unit_words).
  3. Variants: pink difference for the first emphasis phrase, glitch entrance on the opening caption and on
     the caption after each script word, blur-in on the middle script words, one behind-subject depth word,
     punch-in zooms on script words with a settle-out after them.
"""
import argparse
import json
import os
import re

STOPWORDS = {
    'nl': set('''de het een en of maar want dus dat die dit deze wat wie waar wanneer hoe om te van voor met
        naar op in uit aan bij tot door over onder als dan ook nog al wel niet geen zo er hier daar
        ik je jij u hij zij ze we wij jullie mij me mijn jouw zijn haar ons onze hun hen men
        is ben bent was waren wordt worden word heb hebt heeft hadden had kan kun kunt kon konden
        zal zult zullen zou zouden moet moeten mag mogen wil wilt gaat ga gaan doe doet even'''.split()),
    'en': set('''a an the and or but so that this these those what who where when how to of for with from in
        on at by as than then also too not no i i'm im me my you your he she it its we us our they them
        their is am are was were be been being have has had do does did can could will would shall should
        may might must just very really yet still even'''.split()),
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

    def dur(u):
        return words[u[-1]]['end'] - words[u[0]]['start']

    changed = True
    while changed:
        changed = False
        for k, u in enumerate(units):
            if dur(u) >= min_s or u[0] in script:
                continue
            prev = units[k - 1] if k > 0 else None
            nxt = units[k + 1] if k + 1 < len(units) else None
            ok_prev = prev is not None and prev[0] not in script and len(prev) + len(u) <= max_w
            ok_next = nxt is not None and nxt[0] not in script and len(nxt) + len(u) <= max_w
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
    args = ap.parse_args(argv)

    style = json.load(open(args.style))
    pr = style.get('plan_rules', {})
    words = json.load(open(args.words))
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
    if has_masks and (bs or behind_w):
        if behind_w is not None:
            picks = [k for k, p in enumerate(plan) if clean(p['text'].split()[0]) in behind_w]
        else:
            picks = s_idx[:bs.get('max_units', 1)] if bs.get('pick') == 'first_script' else []
        for k in picks:
            plan[k]['behind_subject'] = True
            plan[k]['position'] = (bs or {}).get('position', 'default')
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
