"""Word timings by CTC forced alignment (wav2vec2), accurate to ~20 ms.

Whisper's word timestamps are estimates and can be off by a few hundred ms; caption timing needs
the moment each word actually starts. This aligns a known transcript to the audio letter by letter.

Usage: python align_words.py AUDIO_OR_VIDEO "transcript text" OUT.json [model]
Default model is Dutch; pass an English one (e.g. jonatasgrosman/wav2vec2-large-xlsr-53-english) for English.
"""
import json
import re
import subprocess
import sys

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SR = 16000


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).copy()
    return x / (np.abs(x).max() + 1e-9) * 0.9


def viterbi(emission, tokens, blank):
    """Standard CTC trellis (as in the torchaudio forced-alignment tutorial), in numpy."""
    T, J = emission.shape[0], len(tokens)
    trellis = np.full((T + 1, J + 1), -np.inf)
    trellis[0, 0] = 0
    trellis[1:, 0] = np.cumsum(emission[:, blank])
    for t in range(T):
        stay = trellis[t, 1:] + emission[t, blank]
        move = trellis[t, :-1] + emission[t, tokens]
        trellis[t + 1, 1:] = np.maximum(stay, move)
    # backtrack
    path, j = [], J
    t = int(np.argmax(trellis[:, J]))
    while j > 0 and t > 0:
        stay = trellis[t - 1, j] + emission[t - 1, blank]
        move = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        if move > stay:
            path.append((j - 1, t - 1, float(np.exp(emission[t - 1, tokens[j - 1]]))))
            j -= 1
        t -= 1
    return path[::-1]       # (token index, frame, prob) for each token's emitting frame(s)


def main(media, text, out_path, model_name="jonatasgrosman/wav2vec2-large-xlsr-53-dutch"):
    proc = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
    audio = load_audio(media)
    with torch.inference_mode():
        inputs = proc(audio, sampling_rate=SR, return_tensors="pt")
        logits = model(inputs.input_values).logits[0]
    emission = torch.log_softmax(logits, dim=-1).numpy()
    frame_s = len(audio) / SR / emission.shape[0]
    print("model hears:", proc.batch_decode(logits.argmax(-1)[None])[0])

    vocab = proc.tokenizer.get_vocab()
    blank = proc.tokenizer.pad_token_id
    words = [w for w in re.sub(r"[^\w' ]", " ", text.lower()).split() if w]
    tokens, owner = [], []
    for wi, w in enumerate(words):
        for ch in w:
            if ch in vocab:
                tokens.append(vocab[ch])
                owner.append(wi)
        if wi < len(words) - 1 and "|" in vocab:
            tokens.append(vocab["|"])
            owner.append(-1)
    path = viterbi(emission, tokens, blank)
    spans = {}
    for ti, frame, prob in path:
        wi = owner[ti]
        if wi < 0:
            continue
        s = spans.setdefault(wi, [frame, frame, []])
        s[0], s[1] = min(s[0], frame), max(s[1], frame)
        s[2].append(prob)
    out = []
    for wi, w in enumerate(words):
        f0, f1, probs = spans[wi]
        out.append({"word": w, "start": round(f0 * frame_s, 3), "end": round((f1 + 1) * frame_s, 3),
                    "score": round(float(np.mean(probs)), 3)})
    json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
    for o in out:
        print(f'{o["start"]:6.3f}-{o["end"]:6.3f}  {o["score"]:.2f}  {o["word"]}')


if __name__ == "__main__":
    main(*sys.argv[1:])
