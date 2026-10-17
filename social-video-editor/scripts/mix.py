"""Mix the voice and the effects track under the captioned video.

    voice:   clip audio -> aresample -> highpass -> fade in/out at the clip's edges -> volume (voice_lufs -
             measured loudness) -> apad
    effects: effects track -> aresample -> apad
    amix (normalize=0, duration=longest) -> alimiter (level off, latency compensation ON) -> atrim to the
    video duration -> aresample; AAC; the video stream is copied.

Lessons kept from the lab: the voice gets a measured fixed gain, not single-pass loudnorm (which cut a
short clip's audio to 1.8 s and resampled it to 192 kHz); the limiter must compensate its latency or every
sound lands 5 ms late; no -shortest (it cut the last frames): both inputs are padded and the result is
trimmed to exactly the video's duration. The voice is cut sample-accurately at the trim points, so it fades
in over a few ms and out over a few tens of ms (style mix.voice_fade): the video starts and ends in silence
and loops without a tick, and a phone's stop-tap in the last frames is faded out, not cut off mid-sound.
When the start was moved back so the first caption's shutter plays in full (Clip.voice_start_frame), the
voice is silent over those frames and fades in where it did before the move.
"""
from common import PipelineError, loudness, run


def voice_fade_in(cfg, voice_start_s):
    """The voice's fade-in: at the clip's start, or at voice_start_s when the start was moved back for the first
    caption's shutter (ffmpeg afade: silence before the fade, so those frames play without the voice)."""
    st = f"st={voice_start_s:.6f}:" if voice_start_s > 0 else ""
    return f"afade=t=in:{st}d={cfg['voice_fade']['in_s']}:curve={cfg['voice_fade']['curve']}"


def voice_gain(voice_wav, cfg, voice_start_s=0.0):
    """Gain (dB) that brings the high-passed voice to the style's voice loudness."""
    measured = loudness(voice_wav, f"aresample={cfg['sample_rate']},highpass=f={cfg['voice_highpass_hz']}"
                        + (f",{voice_fade_in(cfg, voice_start_s)}" if voice_start_s > 0 else ""))["lufs"]
    if measured == float("-inf"):
        raise PipelineError("the clip's audio is silent")
    return cfg["voice_lufs"] - measured, measured


def graph(cfg, gain_db, duration, voice="1:a", effects="2:a", voice_start_s=0.0):
    """The mix as an ffmpeg filter graph with output [a]. The verification runs the same graph with
    silence as the effects, to know exactly what the voice alone sounds like after the limiter."""
    sr, lim, fade = cfg["sample_rate"], cfg["limiter"], cfg["voice_fade"]
    return (f"[{voice}]aresample={sr},highpass=f={cfg['voice_highpass_hz']},"
            f"{voice_fade_in(cfg, voice_start_s)},"
            f"afade=t=out:st={duration - fade['out_s']:.6f}:d={fade['out_s']}:curve={fade['curve']},"
            f"volume={gain_db:.3f}dB,apad[v];"
            f"[{effects}]aresample={sr},apad[f];"
            f"[v][f]amix=inputs=2:normalize=0:duration=longest,"
            f"alimiter=limit={lim['limit']}:level={int(lim['level'])}:latency={int(lim['latency'])},"
            f"atrim=end_sample={int(round(duration * sr))},aresample={sr}[a]")


def mix(video, voice_wav, sfx_wav, out_path, cfg, duration, voice_start_s=0.0):
    gain, measured = voice_gain(voice_wav, cfg, voice_start_s)
    g = graph(cfg, gain, duration, voice_start_s=voice_start_s)
    run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", video, "-i", voice_wav, "-i", sfx_wav,
         "-filter_complex", g, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
         "-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"], "-ar", str(cfg["sample_rate"]),
         "-movflags", "+faststart", out_path], "mix")
    return dict(voice_measured_lufs=measured, voice_gain_db=round(gain, 3), voice_start_s=round(voice_start_s, 6), filter=g)


def voice_alone(voice_wav, out_wav, cfg, gain_db, duration, voice_start_s=0.0):
    """The voice through the identical chain and limiter with silent effects (float WAV, no codec)."""
    run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", voice_wav,
         "-f", "lavfi", "-i", f"anullsrc=r={cfg['sample_rate']}:cl=stereo",
         "-filter_complex", graph(cfg, gain_db, duration, "0:a", "1:a", voice_start_s), "-map", "[a]",
         "-c:a", "pcm_f32le", out_wav], "voice rebuild")
