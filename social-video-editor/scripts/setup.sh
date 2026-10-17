#!/usr/bin/env bash
# Create the Python environment for edit_video.py (CPU-only torch) and print its python path.
# Idempotent: an environment that already has exactly requirements.txt installed is left alone, so
# callers can always run   PY="$(bash scripts/setup.sh)"   and then   "$PY" scripts/edit_video.py ...
# The venv goes to $SOCIAL_VIDEO_EDITOR_VENV (default ~/.cache/social-video-editor/venv). Models
# (faster-whisper, wav2vec2) are downloaded by Hugging Face on first use and cached.
# Everything but the final python path goes to stderr.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SOCIAL_VIDEO_EDITOR_VENV:-$HOME/.cache/social-video-editor/venv}"
PYTHON="${PYTHON:-python3}"
REQ="$HERE/requirements.txt"
STAMP="$VENV/.requirements.sha256"

for tool in "$PYTHON" ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || { echo "setup: $tool not found" >&2; exit 1; }
done
"$PYTHON" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 13))' \
  || echo "setup: warning: built and tested with Python 3.13; pinned wheels may not exist for $("$PYTHON" -V)" >&2

want="$(sha256sum "$REQ" | cut -d' ' -f1)"
if [ -x "$VENV/bin/python" ] && [ "$(cat "$STAMP" 2>/dev/null || true)" = "$want" ]; then
  echo "$VENV/bin/python"
  exit 0
fi

echo "setup: creating $VENV" >&2
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip >&2
"$VENV/bin/python" -m pip install --quiet -r "$REQ" >&2
# the caption renderer needs Pillow with raqm (complex text layout, variable-font axes)
"$VENV/bin/python" -c "import faster_whisper, torch, transformers, matplotlib, numpy
from PIL import features
assert features.check('raqm'), 'Pillow was built without raqm'" >&2
echo "$want" > "$STAMP"
echo "$VENV/bin/python"
