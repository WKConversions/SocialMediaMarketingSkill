#!/usr/bin/env python3
"""Regression cases for edit_video.py: run each case end to end and check how it comes out.

    python regression.py [--out DIR] [CASE ...]        (default: every case, into test_out/regression)

Run it with the pipeline's python (scripts/setup.sh prints it). The footage is not in git; the paths are
relative to the repository root (lab/caption-style-01/footage/...). Each case writes OUT/<case>/ (the
pipeline's own output plus log.txt) and prints PASS or FAIL with the reasons; the exit code is the number of
failed cases.

  nl-default            the client's Dutch test clip, automatic trim: every check passes
  en-tos-24-30.5        Tears of Steel 24-30.5 s (English, 1920x1080). "Passions" and "Space" are lopsided
                        script words whose thin hairlines vanish in the visible-ink box, which once measured them
                        3-7.5 px off centre: they must measure centred. "Tom" starts 0.07 s after --start, so
                        caption 1 would appear on frame 1 and cut its shutter's first click off: the start moves
                        back (one frame, to 23.967 s; the voice silent over it and aligned from where it starts:
                        "Tom" is barely audible under the film's music, and aligned together with the extra frames
                        it landed anywhere from their first frame to its real start) and every check passes
  nl-start-1.12         --start 1.12 --end 3.2 on the Dutch clip: "Dit" starts 2 ms after the start, caption 1
                        would be on frame 0: the start moves back (about two frames, from source frame 34; the voice
                        silent over them and still starting on frame 34), every check passes
  nl-speech-at-start    the Dutch clip cut at 1.12 s (made here with ffmpeg), so speech starts on its first frame
                        and there is nothing earlier to move back to: nothing is rendered (exit 2) and plan.json
                        names the problem
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
REPO = os.path.dirname(SKILL)
NL = "lab/caption-style-01/footage/user/IMG_6574.mov"
TOS = "lab/caption-style-01/footage/source/tears_of_steel/tears_of_steel_720p.mov"


def check(name, ok, why, failures):
    if not ok:
        failures.append(f"{name}: {why}")


def all_checks_pass(r, f):
    if r is None:
        f.append("no report.json")
        return
    for c in r["checks"]:
        check(c["name"], c["ok"], "failed", f)


def first_caption_on_frame_2_or_later(r, plan, f):
    frame = r["captions"][0]["detected_first_frame"] if r else None
    check("first caption frame", frame is not None and frame >= 2, f"caption 1 on frame {frame}", f)


def moved_back(plan, requested_frame, f):
    """The start moved back (logged in plan.json notes), the voice silent over the added frames and starting on
    the requested source frame."""
    c = plan["clip"]
    check("trim moved back", bool(plan.get("notes")) and c.get("voice_start_frame", 0) >= 1
          and c["start_frame"] + c["voice_start_frame"] == requested_frame,
          f"start_frame {c['start_frame']}, voice_start_frame {c.get('voice_start_frame')} (the voice should start on "
          f"source frame {requested_frame}), notes {plan.get('notes')}", f)


def centred(r, texts, f):
    caps = {c["text"]: c for c in r["captions"]} if r else {}
    for t in texts:
        c = caps.get(t)
        check(f"{t!r} centred", c is not None and c["placement_worst_shift_px"] is not None
              and max(c["placement_worst_shift_px"]) <= 2, f"{c and c['placement_worst_shift_px']}", f)


def expect_nl_default(code, plan, r, f):
    check("exit code", code == 0, code, f)
    all_checks_pass(r, f)
    first_caption_on_frame_2_or_later(r, plan, f)


def expect_en(code, plan, r, f):
    check("exit code", code == 0, code, f)
    all_checks_pass(r, f)
    first_caption_on_frame_2_or_later(r, plan, f)
    moved_back(plan, 720, f)
    centred(r, ["Passions", "Space"], f)


def expect_nl_start(code, plan, r, f):
    check("exit code", code == 0, code, f)
    all_checks_pass(r, f)
    first_caption_on_frame_2_or_later(r, plan, f)
    moved_back(plan, 34, f)


def expect_refused(code, plan, r, f):
    check("exit code", code == 2, code, f)
    check("not rendered", r is None, "report.json written", f)
    probs = (plan or {}).get("problems", [])
    check("plan names the problem", any("caption 1" in p and "no earlier frame" in p for p in probs), probs, f)


def cut_at_start(out):
    """The Dutch clip from 1.12 s on (speech on its first frame), as a short local file."""
    path = f"{out}/IMG_6574_from_1.12.mov"
    if not os.path.exists(path):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", "1.12", "-i", f"{REPO}/{NL}", "-t", "2.5",
                        "-vf", "scale=1080:1920", "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-c:a", "pcm_s16le", path], check=True)
    return path


CASES = {
    "nl-default": (lambda out: f"{REPO}/{NL}", [], expect_nl_default),
    "en-tos-24-30.5": (lambda out: f"{REPO}/{TOS}", ["--start", "24", "--end", "30.5"], expect_en),
    "nl-start-1.12": (lambda out: f"{REPO}/{NL}", ["--start", "1.12", "--end", "3.2"], expect_nl_start),
    "nl-speech-at-start": (cut_at_start, [], expect_refused),
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="*", help=f"cases to run (default all: {', '.join(CASES)})")
    ap.add_argument("--out", default=f"{SKILL}/test_out/regression")
    a = ap.parse_args()
    failed = 0
    for name in a.cases or list(CASES):
        source, args, expect = CASES[name]
        out = os.path.abspath(f"{a.out}/{name}")
        os.makedirs(out, exist_ok=True)
        for stale in ("report.json", "plan.json", "final.mp4"):
            if os.path.exists(f"{out}/{stale}"):
                os.remove(f"{out}/{stale}")
        cmd = [sys.executable, f"{HERE}/edit_video.py", source(out), "--style", "prime", "--out", out] + args
        t = time.time()
        with open(f"{out}/log.txt", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        failures = []
        expect(code, load(f"{out}/plan.json"), load(f"{out}/report.json"), failures)
        failed += bool(failures)
        print(f"{'PASS' if not failures else 'FAIL'} {name} (exit {code}, {time.time() - t:.0f} s) -> {out}", flush=True)
        for x in failures:
            print(f"    {x}", flush=True)
    return failed


if __name__ == "__main__":
    sys.exit(main())
