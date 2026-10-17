"""Mix the voice and the effects track under the captioned video.

    voice:   clip audio -> aresample -> highpass -> volume (voice_lufs - measured loudness) -> apad
    effects: effects track -> aresample -> apad
    amix (normalize=0, duration=longest) -> alimiter (level off, latency compensation ON) -> atrim to the
    video duration -> aresample; AAC; the video stream is copied.

Lessons kept from the lab: the voice gets a measured fixed gain, not single-pass loudnorm (which cut a
short clip's audio to 1.8 s and resampled it to 192 kHz); the limiter must compensate its latency or every
sound lands 5 ms late; no -shortest (it cut the last frames): both inputs are padded and the result is
trimmed to exactly the video's duration.
"""
from common import PipelineError, loudness, run


def voice_chain(cfg, gain_db):
    return f"aresample={cfg['sample_rate']},highpass=f={cfg['voice_highpass_hz']},volume={gain_db:.3f}dB"


def voice_gain(voice_wav, cfg):
    """Gain (dB) that brings the high-passed voice to the style's voice loudness."""
    measured = loudness(voice_wav, f"aresample={cfg['sample_rate']},highpass=f={cfg['voice_highpass_hz']}")["lufs"]
    if measured == float("-inf"):
        raise PipelineError("the clip's audio is silent")
    return cfg["voice_lufs"] - measured, measured


def mix(video, voice_wav, sfx_wav, out_path, cfg, duration):
    gain, measured = voice_gain(voice_wav, cfg)
    lim = cfg["limiter"]
    graph = (f"[1:a]{voice_chain(cfg, gain)},apad[v];"
             f"[2:a]aresample={cfg['sample_rate']},apad[f];"
             f"[v][f]amix=inputs=2:normalize=0:duration=longest,"
             f"alimiter=limit={lim['limit']}:level={int(lim['level'])}:latency={int(lim['latency'])},"
             f"atrim=end_sample={int(round(duration * cfg['sample_rate']))},aresample={cfg['sample_rate']}[a]")
    run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", video, "-i", voice_wav, "-i", sfx_wav,
         "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
         "-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"], "-ar", str(cfg["sample_rate"]),
         "-movflags", "+faststart", out_path], "mix")
    return dict(voice_measured_lufs=measured, voice_gain_db=round(gain, 3), filter=graph)
