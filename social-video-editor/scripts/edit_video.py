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
  --start/--end S    trim window in seconds of the source, instead of the automatic one around the words

Writes OUT_DIR/final.mp4, OUT_DIR/report.json, OUT_DIR/proof_shutter_alignment.png, OUT_DIR/proof_timeline.png
and intermediates in OUT_DIR/work/. Exit code: 0 = every check passed; 1 = the video was made but a rule
is broken (see report.json); 2 = the pipeline could not run.
"""
import argparse
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
    log("transcribing (faster-whisper %s)" % settings["transcribe"]["model"])
    text, lang, prob = transcribe(audio, settings["transcribe"], None if args.lang == "auto" else args.lang)
    log(f"  {lang} ({prob:.2f}): {text}")
    models = settings["align"]["models"]
    if lang not in models:
        raise PipelineError(f"no alignment model for language {lang!r} (have: {', '.join(models)}); pass --lang")
    words = text.split()
    if not words:
        raise PipelineError("no speech found")
    al = dict(model_name=models[lang], chunk_s=settings["align"]["chunk_s"], phases=settings["align"]["phases"])
    log("aligning the transcript to the source (%s)" % models[lang])
    src_words = align(audio, words, **al)
    save_json(f"{work}/words_source.json", dict(language=lang, text=text, words=src_words))
    clip = Clip.make(info, settings, args.start, args.end, src_words[0]["start"], src_words[-1]["end"])
    kept = [w["word"] for w in src_words
            if clip.start_s <= (w["start"] + w["end"]) / 2 < clip.end_frame / clip.fps]
    log(f"trim: source frames {clip.start_frame}-{clip.end_frame - 1} ({clip.n_frames} frames, {clip.duration:.3f} s) "
        f"-> {clip.width}x{clip.height} @ {clip.fps}")
    log("aligning the transcript to the clip")
    seg = audio[int(round(clip.start_s * SR)):int(round(clip.end_frame / clip.fps * SR))]
    clip_words = align(seg, kept, **al)
    forced = {w.strip().lower() for w in args.emphasis.split(",") if w.strip()} if args.emphasis else None
    plan = build(clip_words, lang, text, style, args.style, clip, forced)
    plan["models"] = dict(transcribe=settings["transcribe"]["model"], align=models[lang])
    return plan, clip, models[lang]


def main(argv=None):
    args = parse_args(argv)
    from plan import describe, normalise

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
    save_json(f"{out}/plan.json", plan)
    log(f"plan -> {out}/plan.json\n" + describe(plan))
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
    fx = sfx.build(plan, style, f"{work}/sfx.wav", f"{work}/sfx_shutter_only.wav")
    log(f"effects track: {len(fx['placed'])} shutters, {sum('hit' in p for p in fx['placed'])} hits, peak {fx['peak_dbfs']} dBFS")
    final = f"{out}/final.mp4"
    mixed = mix.mix(f"{work}/captions.mp4", voice_wav, f"{work}/sfx.wav", final, style["mix"], clip.duration)
    log(f"mix: voice {mixed['voice_measured_lufs']} LUFS + {mixed['voice_gain_db']} dB -> {final}")
    log("verifying the final file")
    report = verify(final, clip, plan, style, settings, work, out, voice_wav, f"{work}/sfx_shutter_only.wav",
                    mixed, align_model)
    report.update(input=info["path"], style=args.style, plan=f"{out}/plan.json", render=rend, sfx=fx,
                  seconds=round(time.time() - T0, 1))
    save_json(f"{out}/report.json", report)
    log(f"{'PASSED' if report['ok'] else 'FAILED'}: {sum(c['ok'] for c in report['checks'])}/{len(report['checks'])} "
        f"checks -> {out}/report.json")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
