"""Mix v17 with the skill's own mix code (social-video-editor/scripts/mix.py): voice to the Prime level, the effects
track, and the client's saved music (music/library.json "tyler-gone-gone": first sound on frame 0, level-matched to
the part of the track used, ducked under the voice, 1.2 s fade-out, total to -14 LUFS). Also writes the clean voice
on the same timeline (for re-aligning) and the music-only stem (to report its level under the voice).
Usage: python3 mix_v15.py VIDEO_NOAUDIO.mp4 OUT.mp4
"""
import json
import subprocess
import sys

sys.path.insert(0, "/home/user/SocialMediaMarketingSkill/social-video-editor/scripts")
import mix  # noqa: E402
from common import loudness  # noqa: E402

E = "/home/user/SocialMediaMarketingSkill/lab/edit-07"
SKILL = "/home/user/SocialMediaMarketingSkill/social-video-editor"


def main(video, out):
    cfg = json.load(open(f"{SKILL}/styles/prime.json"))["mix"]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                               capture_output=True, text=True).stdout)
    music = mix.load_music("tyler-gone-gone")
    # the level corrects itself to the approved 6.7 dB under the voice (library.json under_voice_db)
    info = mix.mix(video, f"{E}/out/voice.wav", f"{E}/out/v17_sfx.wav", out, cfg, dur, 0.0, music=music)
    mix.voice_alone(f"{E}/out/voice.wav", f"{E}/out/v17_voice_alone.wav", cfg, info["voice_gain_db"], dur)
    bed = dict(music, gain_db=info["music_gain_db"])
    mix.voice_alone(f"{E}/out/voice.wav", f"{E}/out/v17_bed_alone.wav", cfg, info["voice_gain_db"], dur, music=bed,
                    master_db=info["master_db"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{E}/out/v17_bed_alone.wav", "-i", f"{E}/out/v17_voice_alone.wav",
                    "-filter_complex", "[1:a]volume=-1[n];[0:a][n]amix=inputs=2:normalize=0", f"{E}/out/v17_music_stem.wav"], check=True)
    v = loudness(f"{E}/out/v17_voice_alone.wav")["lufs"]
    m = loudness(f"{E}/out/v17_music_stem.wav")["lufs"]
    f = loudness(out)
    print(f"music gain {info['music_gain_db']} dB, master {info['master_db']} dB; voice {v:.1f} LUFS, music {m:.1f} LUFS "
          f"({v - m:.1f} dB under the voice); final {f['lufs']:.1f} LUFS, true peak {f['true_peak_dbtp']:.1f} dBTP -> {out}")
    json.dump({k: v for k, v in info.items() if k != "filter"}, open(f"{E}/out/v17_mix.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
