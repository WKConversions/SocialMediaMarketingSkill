#!/usr/bin/env python3
"""Word-level transcript with faster-whisper -> [{"word", "raw", "start", "end"}] (the build_plan.py input).

    python3 -I transcribe_words.py AUDIO.wav OUT.json [--model medium] [--lang en] [--offset S]
"""
import argparse
import json
import wave

import numpy as np
from faster_whisper import WhisperModel


def load_wav(path):
    """16 kHz mono 16-bit PCM wav -> float32 array (avoids faster-whisper's PyAV decoder)."""
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit('expected 16 kHz mono 16-bit wav (ffmpeg -ac 1 -ar 16000)')
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('out')
    ap.add_argument('--model', default='medium')
    ap.add_argument('--lang', default=None)
    ap.add_argument('--offset', type=float, default=0.0, help='subtract from every time')
    a = ap.parse_args()
    m = WhisperModel(a.model, device='cpu', compute_type='int8')
    segs, info = m.transcribe(load_wav(a.audio), language=a.lang, word_timestamps=True, vad_filter=False)
    words = []
    for s in segs:
        print(f'[{s.start:6.2f}-{s.end:6.2f}] {s.text}')
        for w in s.words:
            raw = w.word.strip()
            words.append(dict(word=raw.strip('.,!?;:"'), raw=raw, start=round(w.start - a.offset, 3), end=round(w.end - a.offset, 3)))
    json.dump(words, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print(len(words), 'words ->', a.out)


if __name__ == '__main__':
    main()
