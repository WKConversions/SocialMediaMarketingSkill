"""Shared helpers: skill paths, style/settings loading, ffmpeg calls, probing and the trimmed clip.

The clip is never written out as an intermediate video: every step that needs its pictures decodes the
source through the same filter chain (Clip.video_filter), so the captions are composited on first-generation
frames and the render, the verification and the proof images all see exactly the same pixels.
"""
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-6

# ffprobe colour-matrix names -> ffmpeg scale filter names
MATRICES = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470", "fcc": "fcc", "smpte240m": "smpte240m"}
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}


class PipelineError(RuntimeError):
    """A step failed or a rule is broken; the CLI prints it and exits non-zero."""


def skill_path(rel):
    return rel if os.path.isabs(rel) else os.path.join(SKILL_DIR, rel)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def load_style(name):
    """A style by name (styles/<name>.json) or by path."""
    path = name if name.endswith(".json") else skill_path(f"styles/{name}.json")
    if not os.path.exists(path):
        known = sorted(f[:-5] for f in os.listdir(skill_path("styles")) if f.endswith(".json"))
        raise PipelineError(f"unknown style {name!r}; known styles: {', '.join(known)}")
    return load_json(path)


def load_settings():
    return load_json(skill_path("scripts/settings.json"))


def hex_rgb(value):
    """'#RRGGBB' -> float32 RGB 0..1."""
    v = value.lstrip("#")
    return np.array([int(v[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def first_frame(t, fps):
    """The first frame on which something that starts at time t is visible (frame k shows time k / fps)."""
    return max(0, math.ceil(t * fps - EPS))


def run(cmd, what=None):
    """Run a command, return stdout bytes; raise PipelineError with the end of stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-12:]
        raise PipelineError(f"{what or cmd[0]} failed:\n" + "\n".join(tail))
    return proc.stdout


def probe(path):
    """Video/audio facts of a media file; width/height are as displayed (rotation applied)."""
    if not os.path.exists(path):
        raise PipelineError(f"input not found: {path}")
    info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], "ffprobe"))
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise PipelineError(f"no video stream in {path}")
    rot = int(float(v.get("tags", {}).get("rotate", 0) or 0))
    for sd in v.get("side_data_list", []) or []:
        rot = int(sd.get("rotation", rot) or rot)
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = (int(x) for x in (v.get("avg_frame_rate") or v["r_frame_rate"]).split("/"))
    duration = float(v.get("duration") or info["format"]["duration"])
    return dict(path=os.path.abspath(path), width=w, height=h, rotation=rot, fps=num / den if den else 0.0,
                frames=int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
                duration=duration, format_duration=float(info["format"].get("duration", duration)),
                codec=v.get("codec_name"), pix_fmt=v.get("pix_fmt"),
                color_space=v.get("color_space", ""), color_primaries=v.get("color_primaries", ""),
                color_transfer=v.get("color_transfer", ""), color_range=v.get("color_range", ""),
                audio=None if a is None else dict(rate=int(a["sample_rate"]), channels=int(a["channels"]),
                                                  codec=a.get("codec_name"),
                                                  duration=float(a.get("duration") or duration)))


def decode_audio(path, sr, channels=1, filters=None):
    """Decode audio with ffmpeg to float32 (n,) mono or (n, channels). Never through PyAV: the installed
    PyAV crashes faster-whisper's own decoder (metadata_errors)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn"]
    if filters:
        cmd += ["-af", filters]
    cmd += ["-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    x = np.frombuffer(run(cmd, "ffmpeg audio decode"), np.float32).copy()
    return x if channels == 1 else x.reshape(-1, channels)


def loudness(path, filters=None):
    """EBU R128 integrated loudness (LUFS), sample peak (dBFS) and true peak (dBTP) of a file's audio."""
    af = (filters + "," if filters else "") + "ebur128=peak=sample+true"
    err = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-nostdin", "-i", path, "-vn", "-af", af, "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    summary = err[err.rfind("Summary:"):]
    if "Summary:" not in err:
        raise PipelineError(f"loudness measurement failed for {path}:\n{err[-800:]}")

    def grab(pattern):
        m = re.search(pattern, summary)
        return float(m.group(1)) if m and m.group(1) != "-inf" else float("-inf")
    return dict(lufs=grab(r"I:\s+(-?[\d.]+|-inf) LUFS"),
                sample_peak_dbfs=grab(r"Sample peak:\s+Peak:\s+(-?[\d.]+|-inf) dBFS"),
                true_peak_dbtp=grab(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf) dBFS"))


@dataclass
class Clip:
    """The trimmed, scaled clip: source frames [start_frame, end_frame) at the output frame rate."""
    source: str
    start_frame: int
    end_frame: int
    fps: int
    width: int
    height: int
    crop: tuple            # (w, h) centre crop of the displayed source to the output aspect
    matrix: str            # source colour matrix for the YUV -> RGB conversion
    range: str             # tv / pc
    audio_rate: int

    @property
    def n_frames(self):
        return self.end_frame - self.start_frame

    @property
    def duration(self):
        return self.n_frames / self.fps

    @property
    def start_s(self):
        return self.start_frame / self.fps

    @classmethod
    def make(cls, info, settings, start_s=None, end_s=None, first_word=None, last_word_end=None):
        """Trim window from explicit times (--start/--end) or around the words (settings.trim)."""
        if info["audio"] is None:
            raise PipelineError("the input has no audio track; there is nothing to caption")
        if info["color_transfer"] in HDR_TRANSFERS:
            raise PipelineError(f"HDR source ({info['color_transfer']}); export or record it as SDR (bt709) first")
        out = settings["output"]
        fps = int(out["fps"])
        total = int(math.floor(min(info["duration"], info["audio"]["duration"]) * fps + EPS))
        tr = settings["trim"]
        if start_s is None:
            start_s = first_word - tr["lead_s"] if first_word is not None else 0.0
        if end_s is None:
            end_s = last_word_end + tr["tail_s"] if last_word_end is not None else total / fps
        start = min(max(0, math.ceil(start_s * fps - EPS)), total - 1)     # at most lead_s of dead air
        end = max(start + 1, min(total, math.ceil(end_s * fps - EPS)))      # at least tail_s after the last word
        w, h = out["vertical"] if info["height"] > info["width"] else out["horizontal"]
        sw, sh = info["width"], info["height"]
        if sw * h > sh * w:                                                  # wider than the output: crop the sides
            cw, ch = int(round(sh * w / h / 2)) * 2, sh
        else:
            cw, ch = sw, int(round(sw * h / w / 2)) * 2
        return cls(source=info["path"], start_frame=start, end_frame=end, fps=fps, width=int(w), height=int(h),
                   crop=(cw, ch), matrix=MATRICES.get(info["color_space"], "bt709"),
                   range="pc" if info["color_range"] == "pc" else "tv", audio_rate=info["audio"]["rate"])

    def to_dict(self):
        d = asdict(self)
        d["crop"] = list(self.crop)
        d.update(start_s=round(self.start_s, 6), duration_s=round(self.duration, 6), frames=self.n_frames)
        return d

    @classmethod
    def from_dict(cls, d):
        keys = cls.__dataclass_fields__
        return cls(**{k: (tuple(v) if k == "crop" else v) for k, v in d.items() if k in keys})

    def video_filter(self, scaler="lanczos"):
        """Frame-accurate trim at the output rate, centre crop, scale, YUV -> RGB with the source's matrix."""
        cw, ch = self.crop
        return (f"fps={self.fps},trim=start_frame={self.start_frame}:end_frame={self.end_frame},setpts=PTS-STARTPTS,"
                f"crop={cw}:{ch},scale={self.width}:{self.height}:flags={scaler}:in_color_matrix={self.matrix}"
                f":in_range={self.range},format=rgb24")

    def frames(self, scaler="lanczos"):
        """Yield the clip's frames as (H, W, 3) uint8 RGB."""
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", self.source, "-an", "-vf", self.video_filter(scaler),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        yield from read_frames(cmd, self.width, self.height, self.n_frames)

    def write_audio(self, path, sr):
        """The clip's own audio (the voice), sample-accurate, as float32 stereo WAV at sr."""
        a = int(round(self.start_s * self.audio_rate))
        b = int(round(self.end_frame / self.fps * self.audio_rate))
        run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", self.source, "-vn",
             "-af", f"atrim=start_sample={a}:end_sample={b},asetpts=PTS-STARTPTS,aresample={sr}",
             "-ac", "2", "-c:a", "pcm_f32le", path], "voice extraction")


def read_frames(cmd, width, height, expected=None):
    """Run an ffmpeg command that writes rgb24 rawvideo to stdout and yield (H, W, 3) frames."""
    size = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            n += 1
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if proc.returncode != 0:
        raise PipelineError(f"ffmpeg decode failed:\n{err[-800:]}")
    if expected is not None and n != expected:
        raise PipelineError(f"decoded {n} frames, expected {expected}")


def file_frames(path, width, height, expected=None):
    """Frames of an encoded bt709 output file as (H, W, 3) uint8 RGB."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-an",
           "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    yield from read_frames(cmd, width, height, expected)
