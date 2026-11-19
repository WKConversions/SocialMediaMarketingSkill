#!/usr/bin/env bash
# The automatic quality checks, in one go, on a draft or a final encode:
#   bash qc.sh film.mp4 [out_dir] [--cuts 120,452]
# 1 motion_check (how much moves), 2 jitter_check (shake), 3 handoff_check (pops, with strips),
# 4 busy_check (too many things on screen), 5 strips (every transition), 6 rhythm (punch_check: scene turns;
# relay_check: breaths between moves), 7 turns (motion_read: the shape of each transition, every drawn frame),
# 8 sound (av_sync: hits against chance, when the film has audio). Read every SHAKE, POP and CROWDED line and
# look at its strip before the final render. The motion probe (spikes, lurches, pops past size) runs on the
# project, not the encode: node scripts/motion_probe.mjs, then python3 scripts/motion_probe.py.
set -u
F="$1"; OUT="${2:-qc}"; shift; [ $# -gt 0 ] && shift
D="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
echo "== 1 motion";  python3 "$D/motion_check.py" "$F" --profile > "$OUT/motion.txt"; head -1 "$OUT/motion.txt"
echo "== 2 shake";   python3 "$D/jitter_check.py" "$F" --png "$OUT/shake.png" "$@" | tee "$OUT/shake.txt" | grep -v "^look" ; S=${PIPESTATUS[0]}
echo "== 3 pops";    python3 "$D/handoff_check.py" "$F" --strips "$OUT/pops" "$@" | tee "$OUT/pops.txt" | tail -4; P=${PIPESTATUS[0]}
echo "== 4 busy";    python3 "$D/busy_check.py" "$F" | tee "$OUT/busy.txt" | tail -1; B=${PIPESTATUS[0]}
echo "== 5 strips";  python3 "$D/strips.py" "$F" "$OUT/strips"
echo "== 6 rhythm";  python3 "$D/punch_check.py" "$F" | tee "$OUT/rhythm.txt"; python3 "$D/relay_check.py" "$F" | tee -a "$OUT/rhythm.txt"
echo "   (REGISTER=ad: a big turn every ~2 s, none more than ~5 s apart, relays under 50%. REGISTER=explainer (default):"
echo "    narrated, one continuous stage: judge with phrase_check (a new visual on every phrase); relays under 70%)"
R=$(python3 -c "
import re,sys; t=open('$OUT/rhythm.txt').read(); f=[]
m=re.search(r'longest ([0-9.]+) s',t); r=re.search(r'relays \(1–6\) ([0-9]+)%',t); g=re.search(r'gaps median ([0-9]+) drawn',t)
if m and float(m.group(1))>5: f.append('a '+m.group(1)+' s stretch without a scene turn')
if r and int(r.group(1))>50: f.append('relays '+r.group(1)+'% (no breaths)')
if g and int(g.group(1))<6: f.append('moves '+g.group(1)+' frames apart')
print('RHYTHM: '+', '.join(f) if f else 'rhythm in range')")
R=$(python3 - "$OUT/rhythm.txt" "${REGISTER:-explainer}" <<'PY'
import re, sys
reg = sys.argv[2]
t = open(sys.argv[1]).read(); f = []
m = re.search(r"longest ([0-9.]+) s", t); r = re.search(r"relays \(1–6\) ([0-9]+)%", t); g = re.search(r"gaps median ([0-9]+) drawn", t)
# targets by register (round M16): short ads and social films turn the scene about every 2 s; narrated explainers
# run on one continuous stage that changes per phrase (7 DemoDuck explainers: up to 63 s without a big turn,
# relays 12–64%), so for them the phrase check is the measure and only an extreme rhythm is flagged.
if reg == "ad":
    if m and float(m.group(1)) > 5: f.append(f"a {m.group(1)} s stretch without a scene turn")
    if r and int(r.group(1)) > 50: f.append(f"relays {r.group(1)}% (no breaths)")
if reg != "ad" and r and int(r.group(1)) > 70: f.append(f"relays {r.group(1)}% (no breaths)")
if g and int(g.group(1)) < 4: f.append(f"moves {g.group(1)} frames apart")
print("RHYTHM: " + ", ".join(f) if f else "rhythm in range")
PY
)
echo "== 7 turns";   python3 "$D/motion_read.py" "$F" "$OUT/turns" --max 12 > "$OUT/turns.txt"; grep -c "cut at" "$OUT/turns.txt" | sed 's/^/   cuts inside moves: /'
echo "   read $OUT/turns/event_*.png: a scene turn accelerates into the cut (ramp ×1.15–1.5) and settles in 1–8 frames"
if ffprobe -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 "$F" | grep -q .; then
echo "== 8 sound";   python3 "$D/av_sync.py" "$F" --hits 16 | tee "$OUT/sound.txt" | tail -2
fi
echo "== result: $([ $S -eq 0 ] && echo 'no shake' || echo 'SHAKE found'), $([ $P -eq 0 ] && echo 'no pops' || echo 'pops to look at'), $([ $B -eq 0 ] && echo 'clear' || echo 'BUSY: fewer, bigger things'), $R — details in $OUT/"
