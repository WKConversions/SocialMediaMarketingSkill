"""Mix for the Wright 1903 classy edit: the voice in blocks on the video timeline, Sinatra "Come Fly With Me" with
volume automation, the soft shutters, mastered to -14 LUFS.

Music automation (out/edit_v19.json music_ramps):
  up     (interludes, the opening line, the end card): each stretch set to 1.5 LU under the voice's own loudness,
         so the music clearly rises when he stops talking and every interlude rises the same amount;
  voice  (while he talks, per block): UNDER_VOICE_DB under the voice, measured the approved way (50 ms blocks with the voice
         above -35 dBFS) and corrected. Deeper than the instrumental Tyler bed (6.7 dB): this song has Sinatra
         singing, and two voices at once are tiring; a gentle cut around 1-3 kHz on the song while he talks
         (where both voices live) keeps the words clear without making the music sound thin.
  ramps  raised-cosine, ~0.4 s, placed in the pauses (down just before his first word, up right after his last).
The limiter compensates its latency (no 5 ms shift); the voice is cut only in its pauses, with short fades.
Usage: python3 mix_v19.py VIDEO_NOAUDIO.mp4 SFX.wav OUT.mp4
"""
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/social-video-editor/scripts")
import mix  # noqa: E402
from common import decode_audio, loudness  # noqa: E402

E = "/home/user/SocialMediaMarketingSkill/lab/edit-09"
SKILL = "/home/user/SocialMediaMarketingSkill/social-video-editor"
SR = 48000
UNDER_VOICE_DB = 12.0          # music under the voice while he speaks
UP_VS_VOICE_LU = -1.5          # interlude music loudness relative to the voice as it sounds (LUFS of the voice track)
VOCAL_CUT = "equalizer=f=2200:t=o:w=1.6:g=-5,equalizer=f=800:t=o:w=1.0:g=-2"


def write_f32(path, x):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", path],
                   input=np.ascontiguousarray(x, np.float32).tobytes(), check=True)


def envelope(ramps, n):
    """1 = music up, 0 = under the voice; raised-cosine ramps."""
    t = np.arange(n) / SR
    e = np.ones(n)
    state = 1.0
    last = 0
    for r in sorted(ramps, key=lambda r: r["t0"]):
        a, b = int(r["t0"] * SR), int(r["t1"] * SR)
        e[last:a] = state
        target = 0.0 if r["to"] == "voice" else 1.0
        p = np.clip((t[a:b] - r["t0"]) / (r["t1"] - r["t0"]), 0, 1)
        s = 0.5 - 0.5 * np.cos(np.pi * p)
        e[a:b] = state + (target - state) * s
        state, last = target, b
    e[last:] = state
    return e


def rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def under_db(voice, music):
    """How far the music sits under the voice while the voice speaks (dB)."""
    blk = int(0.05 * SR)
    k = min(len(voice), len(music)) // blk
    vb = voice[: k * blk].reshape(k, blk)
    mb = music[: k * blk].reshape(k, blk)
    act = 10 * np.log10((vb ** 2).mean(axis=1) + 1e-12) > -35
    return rms_db(vb[act]) - rms_db(mb[act])


def main(video, sfx_wav, out):
    cfg = json.load(open(f"{SKILL}/styles/prime.json"))["mix"]
    ed = json.load(open(f"{E}/out/edit_v19.json"))
    dur = ed["duration"]
    n = int(round(dur * SR))
    song = mix.load_music("sinatra-come-fly-with-me")

    # voice: high-passed, fixed gain to the style's voice loudness, laid out in blocks on the video timeline
    gain, measured = mix.voice_gain(f"{E}/out/voice.wav", cfg)
    v = decode_audio(f"{E}/out/voice.wav", SR, 1, f"highpass=f={cfg['voice_highpass_hz']}") * 10 ** (gain / 20)
    voice = np.zeros(n)
    for b in ed["blocks"]:
        seg = v[int(round(b["v0"] * SR)): int(round(b["v1"] * SR))].copy()
        fi, fo = int(0.005 * SR), int(0.03 * SR)
        seg[:fi] *= np.linspace(0, 1, fi) ** 2
        seg[-fo:] *= np.linspace(1, 0, fo) ** 2
        a = int(round(b["start"] * SR))
        voice[a: a + len(seg)] += seg[: n - a]

    # music: the song from the approved start, plain and with the vocal-range cut, blended by the automation
    trim = f"atrim=start={song['start_offset_s']}:end={song['start_offset_s'] + dur + 0.1:.4f},asetpts=PTS-STARTPTS"
    m = decode_audio(song["path"], SR, 2, trim)[:n]
    m_cut = decode_audio(song["path"], SR, 2, f"{trim},{VOCAL_CUT}")[:n]
    m = np.pad(m, ((0, n - len(m)), (0, 0)))
    m_cut = np.pad(m_cut, ((0, n - len(m_cut)), (0, 0)))
    e = envelope(ed["music_ramps"], n)
    fade = np.ones(n)
    fi = int(0.03 * SR)
    fade[:fi] = np.linspace(0, 1, fi)
    fo = int(ed["music"]["fade_out_s"] * SR)
    fade[-fo:] = 0.5 + 0.5 * np.cos(np.pi * np.linspace(0, 1, fo))

    # sections: every 'up' stretch (opening line, interludes, end card) and every speaking block gets its own gain,
    # because the song's loudness changes a lot (one interlude falls on a passage 4 LU quieter than the others)
    up_mask = e > 0.5
    edges = np.flatnonzero(np.diff(up_mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [n]])
    regions = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if up_mask[a]]
    blocks = [(int(b["start"] * SR), int(b["end"] * SR)) for b in ed["blocks"]]

    def nearest(ranges):
        idx = np.zeros(n, np.int32)
        mids = np.array([(a + b) / 2 for a, b in ranges])
        half = np.array([(b - a) / 2 for a, b in ranges])
        pos = np.arange(n)
        dist = np.abs(pos[:, None] - mids[None, :]) - half[None, :]          # distance to each range (<=0 inside)
        idx[:] = np.argmin(dist, axis=1)
        return idx

    r_idx, b_idx = nearest(regions), nearest(blocks)
    write_f32(f"{E}/out/v19_tmp_voice.wav", np.repeat(voice[:, None], 2, axis=1))
    voice_l = loudness(f"{E}/out/v19_tmp_voice.wav")["lufs"]                  # the voice as it really sounds (~-12.5)
    target_up = voice_l + UP_VS_VOICE_LU
    g_up = np.zeros(len(regions))
    for k, (a, b) in enumerate(regions):                                       # each up stretch to the same loudness
        core = slice(a, b)
        write_f32(f"{E}/out/v19_tmp_up.wav", m[core])
        g_up[k] = float(np.clip(target_up - loudness(f"{E}/out/v19_tmp_up.wav")["lufs"], -12, 6))
    g_voice = np.array([float(np.median(g_up)) - 12.0] * len(blocks))

    def music_track():
        gu, gv = g_up[r_idx], g_voice[b_idx]
        g = gv + e * (gu - gv)
        return (e[:, None] * m + (1 - e[:, None]) * m_cut) * (10 ** (g / 20) * fade)[:, None]

    for _ in range(3):                                                          # each block exactly UNDER_VOICE_DB under
        mt = music_track()
        for k, (a, b) in enumerate(blocks):
            g_voice[k] -= UNDER_VOICE_DB - under_db(voice[a:b], mt[a:b].mean(axis=1))
    mt = music_track()
    under = [round(float(under_db(voice[a:b], mt[a:b].mean(axis=1))), 2) for a, b in blocks]
    u = float(under_db(voice, mt.mean(axis=1)))
    ups = []
    for a, b in regions:
        write_f32(f"{E}/out/v19_tmp_up.wav", mt[a:b])
        ups.append(round(loudness(f"{E}/out/v19_tmp_up.wav")["lufs"], 2))
    l = float(np.mean(ups))

    sfx = decode_audio(sfx_wav, SR, 2)[:n]
    sfx = np.pad(sfx, ((0, n - len(sfx)), (0, 0)))
    total = voice[:, None] + mt + sfx
    write_f32(f"{E}/out/v19_premaster.wav", total)
    write_f32(f"{E}/out/v19_voice_alone.wav", np.repeat(voice[:, None], 2, axis=1))
    write_f32(f"{E}/out/v19_music_stem.wav", mt)
    lim = cfg["limiter"]

    def render(master_db):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", video, "-i", f"{E}/out/v19_premaster.wav", "-filter_complex",
                        f"[1:a]volume={master_db:.3f}dB,alimiter=limit={lim['limit']}:level=0:latency=1,"
                        f"atrim=end_sample={n},aresample={SR}[a]", "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                        "-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"], "-ar", str(SR), "-movflags", "+faststart", out],
                       check=True)

    master = 0.0
    render(master)
    for _ in range(2):
        master += cfg["final_lufs"] - loudness(out)["lufs"]
        render(master)
    f = loudness(out)
    info = dict(voice_measured_lufs=measured, voice_gain_db=round(gain, 3), voice_timeline_lufs=round(voice_l, 2),
                music_up_gain_db=[round(float(x), 2) for x in g_up], music_up_lufs=ups,
                music_voice_gain_db=[round(float(x), 2) for x in g_voice], music_under_voice_db_per_block=under,
                music_under_voice_db=round(u, 2), master_db=round(master, 3), final_lufs=f["lufs"], true_peak_dbtp=f["true_peak_dbtp"])
    json.dump(info, open(f"{E}/out/v19_mix.json", "w"), indent=1)
    print(f"music up to ~{l:.1f} LUFS in the interludes {ups}, under the voice {under} dB; master {master:+.2f} dB; "
          f"final {f['lufs']:.1f} LUFS, true peak {f['true_peak_dbtp']:.1f} dBTP -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
