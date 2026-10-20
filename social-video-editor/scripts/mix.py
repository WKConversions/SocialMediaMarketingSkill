"""Mix the voice, the effects track and (optionally) a music bed under the captioned video.

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
import json
import os

import numpy as np

from common import PipelineError, decode_audio, loudness, run

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_music(name):
    """A track from music/library.json (timing and levels the client approved); the audio file must exist."""
    lib = json.load(open(os.path.join(SKILL, "music", "library.json")))["tracks"]
    if name not in lib:
        raise PipelineError(f"unknown music {name!r}; known: {', '.join(lib)}")
    t = dict(lib[name], name=name)
    t["path"] = os.path.join(SKILL, t["file"])
    if not os.path.exists(t["path"]):
        raise PipelineError(f"music file missing: put {t['title']} by {t['artist']} at {t['path']} "
                            "(audio files are not in git; ask the client for it)")
    return t


def music_gain(music, duration, cfg):
    """Gain (dB) that puts the part of the track this video uses at the approved level relative to the voice.
    Tracks are not equally loud all the way through (this one's first 10 s are 3 dB quieter than the rest), so
    a fixed pre-gain would leave a short video's bed too soft; clamped to pre_gain_db +-6 dB."""
    m, start = music["mix"], music["start_offset_s"]
    if "level_lu_vs_voice" not in m:
        return m["pre_gain_db"]
    seg = loudness(music["path"], f"atrim=start={start}:end={start + duration:.6f}")["lufs"]
    if seg == float("-inf"):
        return m["pre_gain_db"]
    gain = cfg["voice_lufs"] + m["level_lu_vs_voice"] - seg
    return round(max(m["pre_gain_db"] - 6.0, min(m["pre_gain_db"] + 6.0, gain)), 2)


def music_under_voice_db(voice_wav, cfg, gain_db, duration, voice_start_s, music, tmp_prefix):
    """How far the music bed sits under the voice WHILE THE VOICE SPEAKS (dB): the voice alone and voice + bed are
    rendered through the identical chain; the bed is their difference; both are compared in 50 ms blocks where the
    voice is above -35 dBFS. This is how the approved mix was measured (6.7 dB on the Brooklyn Bridge edit)."""
    sr = cfg["sample_rate"]
    v_path, b_path = f"{tmp_prefix}_voice.wav", f"{tmp_prefix}_bed.wav"
    voice_alone(voice_wav, v_path, cfg, gain_db, duration, voice_start_s)
    voice_alone(voice_wav, b_path, cfg, gain_db, duration, voice_start_s, music=music)
    v = decode_audio(v_path, sr, channels=2).mean(axis=1)
    b = decode_audio(b_path, sr, channels=2).mean(axis=1)
    for p in (v_path, b_path):
        os.remove(p)
    n = min(len(v), len(b))
    m = b[:n] - v[:n]
    blk = int(0.05 * sr)
    k = n // blk
    vb, mb = v[: k * blk].reshape(k, blk), m[: k * blk].reshape(k, blk)
    vdb = 10 * np.log10((vb ** 2).mean(axis=1) + 1e-12)
    act = vdb > -35
    if act.sum() < 4:
        return None
    return 10 * np.log10((vb[act] ** 2).mean() + 1e-12) - 10 * np.log10((mb[act] ** 2).mean() + 1e-12)


def music_chain(music, duration, sr, music_in, sidechain):
    """Music bed: first audible sample on frame 0, level-matched gain, ducked under the voice, faded out at the end."""
    m, d = music["mix"], music["mix"]["duck"]
    return (f"[{music_in}]aresample={sr},atrim=start={music['start_offset_s']},asetpts=PTS-STARTPTS,"
            f"volume={music.get('gain_db', m['pre_gain_db'])}dB,afade=t=out:st={duration - m['fade_out_s']:.6f}:d={m['fade_out_s']}[mus];"
            f"[mus][{sidechain}]sidechaincompress=threshold={d['threshold']}:ratio={d['ratio']}:"
            f"attack={d['attack_ms']}:release={d['release_ms']}[md];")


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


def graph(cfg, gain_db, duration, voice="1:a", effects="2:a", voice_start_s=0.0, music=None, music_in="3:a", master_db=0.0):
    """The mix as an ffmpeg filter graph with output [a]. The verification runs the same graph with
    silence as the effects, to know exactly what the voice (and music bed) alone sound like after the limiter.
    With music: the voice is split to key the music's ducking, the bed is mixed in, and master_db brings the
    total to the style's final loudness (without music master_db stays 0, exactly the approved Prime chain)."""
    sr, lim, fade = cfg["sample_rate"], cfg["limiter"], cfg["voice_fade"]
    g = (f"[{voice}]aresample={sr},highpass=f={cfg['voice_highpass_hz']},"
         f"{voice_fade_in(cfg, voice_start_s)},"
         f"afade=t=out:st={duration - fade['out_s']:.6f}:d={fade['out_s']}:curve={fade['curve']},"
         f"volume={gain_db:.3f}dB,apad" + (",asplit=2[v][vsc];" if music else "[v];") +
         f"[{effects}]aresample={sr},apad[f];")
    if music:
        g += music_chain(music, duration, sr, music_in, "vsc")
        g += "[v][f][md]amix=inputs=3:normalize=0:duration=longest,"
    else:
        g += "[v][f]amix=inputs=2:normalize=0:duration=longest,"
    if abs(master_db) > 1e-9:
        g += f"volume={master_db:.3f}dB,"
    g += (f"alimiter=limit={lim['limit']}:level={int(lim['level'])}:latency={int(lim['latency'])},"
          f"atrim=end_sample={int(round(duration * sr))},aresample={sr}[a]")
    return g


def mix(video, voice_wav, sfx_wav, out_path, cfg, duration, voice_start_s=0.0, music=None):
    gain, measured = voice_gain(voice_wav, cfg, voice_start_s)
    under = None
    if music:
        music = dict(music, gain_db=music_gain(music, duration, cfg))
        target = music["mix"].get("under_voice_db")
        if target is not None:
            # the loudness match is a starting point; how loud the bed sounds under THIS voice depends on the speech
            # (density, level, ducking), so measure it while the voice speaks and correct to the approved distance
            first = music_under_voice_db(voice_wav, cfg, gain, duration, voice_start_s, music, out_path + ".music")
            if first is not None:
                lo, hi = music["mix"]["pre_gain_db"] - 9.0, music["mix"]["pre_gain_db"] + 6.0
                music["gain_db"] = round(max(lo, min(hi, music["gain_db"] - (target - first))), 2)
                under = music_under_voice_db(voice_wav, cfg, gain, duration, voice_start_s, music, out_path + ".music")
                under = None if under is None else round(under, 2)

    def render(master_db):
        g = graph(cfg, gain, duration, voice_start_s=voice_start_s, music=music, master_db=master_db)
        run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", video, "-i", voice_wav, "-i", sfx_wav]
            + (["-i", music["path"]] if music else []) +
            ["-filter_complex", g, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
             "-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"], "-ar", str(cfg["sample_rate"]),
             "-movflags", "+faststart", out_path], "mix")
        return g

    master = 0.0
    g = render(master)
    if music and music["mix"].get("master_to_final_lufs"):
        # the bed adds loudness: one correction pass brings the total to the style's final target
        first = loudness(out_path)["lufs"]
        master = max(-6.0, min(6.0, cfg["final_lufs"] - first))
        g = render(master)
    return dict(voice_measured_lufs=measured, voice_gain_db=round(gain, 3), voice_start_s=round(voice_start_s, 6), filter=g,
                music=music["name"] if music else None, music_gain_db=music["gain_db"] if music else None,
                music_under_voice_db=under,
                master_db=round(master, 3))


def voice_alone(voice_wav, out_wav, cfg, gain_db, duration, voice_start_s=0.0, music=None, master_db=0.0):
    """The voice (plus the music bed when given) through the identical chain and limiter with silent effects
    (float WAV, no codec). Without music it is the clean voice used for re-alignment."""
    run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", voice_wav,
         "-f", "lavfi", "-i", f"anullsrc=r={cfg['sample_rate']}:cl=stereo"] + (["-i", music["path"]] if music else []) +
        ["-filter_complex", graph(cfg, gain_db, duration, "0:a", "1:a", voice_start_s, music=music, music_in="2:a",
                                  master_db=master_db), "-map", "[a]",
         "-c:a", "pcm_f32le", out_wav], "voice rebuild")
