#!/usr/bin/env python3
"""Edit a talking-to-camera social video in a caption style, end to end, and verify the finished file.

    python edit_video.py INPUT.mov --style prime --out OUT_DIR [--lang auto|nl|en] [--plan-only]
                         [--plan PLAN.json] [--emphasis "word,word"] [--start S --end S]

Steps: probe -> transcribe (faster-whisper) -> forced alignment (wav2vec2) -> trim (frame-accurate, 30 fps,
1080x1920 / 1920x1080, bt709) -> caption plan, written to OUT_DIR/plan.json -> [--plan-only stops here so
the plan can be edited] -> render captions -> effects track -> mix -> verify -> OUT_DIR/final.mp4.

  --plan PLAN.json   start from an (edited) plan: skips transcription, alignment and planning and uses the
                     plan's trim window; the normalised plan is written to OUT_DIR/plan.json
  --emphasis a,b     these words become the emphasis (script) words instead of the automatic choice
  --start/--end S    trim window in seconds of the source, instead of the automatic one around the words;
                     only this window (plus trim.window_margin_s) is transcribed and aligned

Writes OUT_DIR/final.mp4, OUT_DIR/report.json, OUT_DIR/proof_shutter_alignment.png, OUT_DIR/proof_timeline.png
and intermediates in OUT_DIR/work/. Exit code: 0 = every check passed; 1 = the video was made but a rule
is broken (see report.json); 2 = the pipeline could not run, or the plan needs a look before rendering
(plan.json is written and lists the problems): a caption too short for the style, or a transcript or
alignment that looks unreliable (settings.json "quality"). Fix the plan, or run again with --lang, then
render it with --plan.
"""
import argparse
import math
import os
import sys
import time

from common import Clip, PipelineError, decode_audio, load_json, load_settings, load_style, probe, save_json

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("input")
    ap.add_argument("--style", required=True, help="style name (styles/<name>.json) or path")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--lang", default="auto", help="auto (detect), nl or en")
    ap.add_argument("--plan-only", action="store_true", help="stop after writing plan.json")
    ap.add_argument("--plan", help="render from this plan instead of planning")
    ap.add_argument("--emphasis", help='comma-separated emphasis words, e.g. "testvideo,editen,willen"')
    ap.add_argument("--start", type=float, help="trim start (s, source time)")
    ap.add_argument("--end", type=float, help="trim end (s, source time)")
    a = ap.parse_args(argv)
    if a.plan and (a.emphasis or a.start is not None or a.end is not None or a.plan_only):
        ap.error("--plan cannot be combined with --emphasis/--start/--end/--plan-only: edit the plan instead")
    return a


def make_plan(args, info, style, settings, work):
    """Transcribe, align, trim and plan. Returns (plan, clip, alignment model)."""
    from align import align
    from plan import build
    from transcribe import SR, transcribe

    audio = decode_audio(info["path"], SR)
    # with an explicit trim window only that window (plus a margin) is transcribed and aligned
    margin = settings["trim"]["window_margin_s"]
    w0 = 0 if args.start is None else max(0, int((args.start - margin) * SR))
    w1 = len(audio) if args.end is None else min(len(audio), int(math.ceil((args.end + margin) * SR)))
    if w1 - w0 < SR // 10:
        raise PipelineError(f"the trim window --start {args.start} --end {args.end} holds no audio")
    window = audio[w0:w1]
    if w0 or w1 < len(audio):
        log(f"transcribing only {w0 / SR:.2f}-{w1 / SR:.2f} s of the source (--start/--end + {margin} s)")
    log("transcribing (faster-whisper %s)" % settings["transcribe"]["model"])
    text, lang, prob = transcribe(window, settings["transcribe"], None if args.lang == "auto" else args.lang)
    log(f"  {lang} ({prob:.2f}): {text}")
    models = settings["align"]["models"]
    if lang not in models:
        raise PipelineError(f"no alignment model for language {lang!r} (have: {', '.join(models)}); pass --lang")
    words = text.split()
    if not words:
        raise PipelineError("no speech found")
    al = dict(model_name=models[lang], chunk_s=settings["align"]["chunk_s"], phases=settings["align"]["phases"])
    log("aligning the transcript to the source (%s)" % models[lang])
    src_words = [dict(w, start=round(w["start"] + w0 / SR, 3), end=round(w["end"] + w0 / SR, 3))
                 for w in align(window, words, **al)]
    save_json(f"{work}/words_source.json", dict(language=lang, language_probability=round(prob, 3), text=text,
                                                window_s=[w0 / SR, w1 / SR], words=src_words))
    clip = Clip.make(info, settings, args.start, args.end, src_words[0]["start"], src_words[-1]["end"])
    kept = kept_from(src_words, clip)
    if not kept:
        raise PipelineError("no spoken word lies in the trim window")
    log(f"trim: source frames {clip.start_frame}-{clip.end_frame - 1} ({clip.n_frames} frames, {clip.duration:.3f} s) "
        f"-> {clip.width}x{clip.height} @ {clip.fps}")
    log("aligning the transcript to the clip")
    seg = audio[int(round(clip.start_s * SR)):int(round(clip.end_frame / clip.fps * SR))]
    clip_words = align(seg, [w["word"] for w in kept], **al)
    forced = {w for w in args.emphasis.split(",") if w.strip()} if args.emphasis else None
    plan = build(clip_words, lang, text, style, args.style, clip, forced)
    plan["models"] = dict(transcribe=settings["transcribe"]["model"], align=models[lang])
    plan["quality"], problems = transcript_quality(prob, clip_words, kept, clip, settings["quality"])
    if problems:
        plan["problems"] = plan.get("problems", []) + problems
    return plan, clip, models[lang]


def kept_from(src_words, clip):
    """The source-window words whose middle lies in the clip (the words aligned again on the clip)."""
    return [w for w in src_words if clip.start_s <= (w["start"] + w["end"]) / 2 < clip.end_frame / clip.fps]


def transcript_quality(prob, clip_words, kept, clip, cfg):
    """Is the transcript and its timing trustworthy enough to render? Whisper's language probability, the mean
    CTC score of the clip alignment, and how far the source-window and clip alignments disagree about the
    words' starts. Returns (measurements, problems)."""
    disagree = max(abs(s["start"] - clip.start_s - c["start"]) for s, c in zip(kept, clip_words))
    q = dict(language_probability=round(prob, 3),
             mean_alignment_score=round(sum(w["score"] for w in clip_words) / len(clip_words), 3),
             max_alignment_disagreement_ms=round(disagree * 1000, 1))
    problems = []
    if prob < cfg["min_language_probability"]:
        problems.append(f"Whisper is unsure of the language (probability {prob:.2f} < {cfg['min_language_probability']}): "
                        "the transcript may be wrong; check it, or run again with --lang")
    if q["mean_alignment_score"] < cfg["min_mean_alignment_score"]:
        problems.append(f"the words fit the audio badly (mean alignment score {q['mean_alignment_score']} < "
                        f"{cfg['min_mean_alignment_score']}): the transcript or the language is probably wrong")
    if disagree > cfg["max_alignment_disagreement_frames"] / clip.fps:
        problems.append(f"aligned on the source and on the clip, a word's start differs by {disagree * 1000:.0f} ms "
                        f"(more than {cfg['max_alignment_disagreement_frames']} frame): the timing is unreliable")
    return q, problems


def main(argv=None):
    args = parse_args(argv)
    from plan import describe, normalise, pauses_not_cut, spoken_words

    style = load_style(args.style)
    settings = load_settings()
    out = os.path.abspath(args.out)
    work = f"{out}/work"
    os.makedirs(work, exist_ok=True)
    info = probe(args.input)
    log(f"probe: {info['width']}x{info['height']} {info['fps']:.3f} fps {info['duration']:.3f} s {info['codec']} "
        f"{info['color_transfer'] or '?'}; audio {info['audio']}")

    if args.plan:
        plan = load_json(args.plan)
        for note in normalise(plan, style):
            log("plan: " + note)
        clip = Clip.from_dict(plan["clip"])
        if os.path.basename(clip.source) != os.path.basename(info["path"]):
            raise PipelineError(f"the plan was made for {clip.source}, not {info['path']}")
        clip.source = plan["clip"]["source"] = info["path"]
        align_model = plan.get("models", {}).get("align") or settings["align"]["models"][plan["language"]]
    else:
        plan, clip, align_model = make_plan(args, info, style, settings, work)
    plan["pauses_not_cut"] = pauses_not_cut(spoken_words(plan)[0], settings["editing"])
    save_json(f"{out}/plan.json", plan)
    log(f"plan -> {out}/plan.json\n" + describe(plan))
    if plan.get("quality"):
        log("  transcript quality: " + ", ".join(f"{k} {v}" for k, v in plan["quality"].items()))
    if clip.tonemap:
        log(f"  HDR source ({clip.tonemap['transfer']}): tone-mapped to bt709 SDR")
    for p in plan["pauses_not_cut"]:
        log(f"  pause {p['length_s']:.2f} s between {p['after']!r} and {p['before']!r}: {p['status']}")
    if plan.get("problems"):
        raise PipelineError(f"not rendered: check {out}/plan.json and run again with --plan (or, for a transcript "
                            "problem, with --lang):\n  " + "\n  ".join(plan["problems"]))
    if args.plan_only:
        return 0

    import mix
    import render_captions
    import sfx
    from verify import verify

    sr = style["mix"]["sample_rate"]
    voice_wav = f"{work}/voice.wav"
    clip.write_audio(voice_wav, sr)
    log("rendering captions")
    rend = render_captions.render(clip, plan, style, f"{work}/captions.mp4", settings["output"]["scaler"])
    log(f"  {rend['frames']} frames")
    fx = sfx.build(plan, style, f"{work}/sfx.wav")
    log(f"effects track: {len(fx['placed'])} shutters, {sum('hit' in p for p in fx['placed'])} hits, peak {fx['peak_dbfs']} dBFS")
    final = f"{out}/final.mp4"
    mixed = mix.mix(f"{work}/captions.mp4", voice_wav, f"{work}/sfx.wav", final, style["mix"], clip.duration)
    log(f"mix: voice {mixed['voice_measured_lufs']} LUFS + {mixed['voice_gain_db']} dB -> {final}")
    log("verifying the final file")
    report = verify(final, clip, plan, style, settings, work, out, voice_wav, mixed, align_model)
    report.update(input=info["path"], style=args.style, plan=f"{out}/plan.json", render=rend, sfx=fx,
                  seconds=round(time.time() - T0, 1))
    save_json(f"{out}/report.json", report)
    for w in report["warnings"]:
        log("warning: " + w)
    log(f"{'PASSED' if report['ok'] else 'FAILED'}: {sum(c['ok'] for c in report['checks'])}/{len(report['checks'])} "
        f"checks -> {out}/report.json")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
