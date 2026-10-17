"""Transcript text and language with faster-whisper.

Only the words and the language are used. Whisper's own word timestamps were 100-400 ms early on the
client's first test, so every timing comes from forced alignment (align.py) instead.
The audio is decoded with ffmpeg (common.decode_audio) and passed as a numpy array: faster-whisper's own
PyAV decoding crashes with the installed PyAV (metadata_errors).
"""
import os

SR = 16000


def transcribe(audio, cfg, language=None):
    """audio: 16 kHz mono float32. Returns (text, language, probability)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(cfg["model"], device="cpu", compute_type=cfg["compute_type"], cpu_threads=os.cpu_count() or 4)
    segments, info = model.transcribe(audio, language=language, beam_size=cfg.get("beam_size", 5),
                                      vad_filter=False, word_timestamps=False)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language, float(info.language_probability)
