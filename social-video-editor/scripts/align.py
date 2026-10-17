"""Word timings by CTC forced alignment (wav2vec2).

Aligns a known transcript to the audio letter by letter: the start of a word is the first model frame that
emits its first letter. One model frame is 20 ms, so a single pass puts every word on a 20 ms grid that
depends on where the audio happens to start. To make the timing independent of that, the alignment is run
`phases` times with the audio shifted by a fraction of a model frame and the times are averaged (about
4 ms resolution with 5 phases). Times within a pass follow the lab convention the approved v11 was timed
with (frame index x audio duration / number of frames).
Long audio goes through the model in overlapping windows (chunk_s) so memory stays bounded; each model
frame is taken from the window where it is furthest from an edge.
"""
import os
import re

import numpy as np

SR = 16000
STRIDE, FIELD = 320, 400          # wav2vec2 feature encoder: hop and receptive field in samples
_MODELS = {}


def load_model(name):
    if name not in _MODELS:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        torch.set_num_threads(os.cpu_count() or 4)
        _MODELS[name] = (Wav2Vec2Processor.from_pretrained(name), Wav2Vec2ForCTC.from_pretrained(name).eval())
    return _MODELS[name]


def n_frames(n_samples):
    return (n_samples - FIELD) // STRIDE + 1


def emissions(audio, proc, model, chunk_s):
    """Log-probabilities (frames, vocab) for the whole audio, computed in windows if it is long."""
    import torch

    def run(x):
        with torch.inference_mode():
            inputs = proc(x, sampling_rate=SR, return_tensors="pt")
            return torch.log_softmax(model(inputs.input_values).logits[0], dim=-1).numpy()

    total = n_frames(len(audio))
    win = int(chunk_s * SR) // STRIDE * STRIDE
    if len(audio) <= win:
        return run(audio)[:total]
    overlap = win // 8 // STRIDE * STRIDE
    starts = list(range(0, len(audio) - overlap, win - overlap))
    half = overlap // STRIDE // 2
    out = None
    for k, s in enumerate(starts):
        e = run(audio[s:s + win])
        if out is None:
            out = np.zeros((total, e.shape[1]), np.float32)
        g0 = s // STRIDE
        lo = 0 if k == 0 else half                                   # skip the frames near the window edges
        hi = min(len(e) if k == len(starts) - 1 else len(e) - half, total - g0)
        out[g0 + lo:g0 + hi] = e[lo:hi]
    return out


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
    path, j = [], J
    t = int(np.argmax(trellis[:, J]))
    while j > 0 and t > 0:
        stay = trellis[t - 1, j] + emission[t - 1, blank]
        move = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        if move > stay:
            path.append((j - 1, t - 1, float(np.exp(emission[t - 1, tokens[j - 1]]))))
            j -= 1
        t -= 1
    return path[::-1]       # (token index, frame, prob) for each token's emitting frame


def align(audio, words, model_name, chunk_s=30, phases=5):
    """audio: 16 kHz mono float32; words: list of strings (as spoken, any case/punctuation).
    Returns [{"word", "start", "end", "score"}] in seconds from the start of the audio, averaged over
    `phases` sub-frame shifts of the model's frame grid."""
    audio = audio / (np.abs(audio).max() + 1e-9) * 0.9
    runs = []
    for k in range(phases):
        d = STRIDE * k // phases
        runs.append([(s + d / SR, e + d / SR, sc) for s, e, sc in _align_once(audio[d:], words, model_name, chunk_s)])
    out = []
    for i, w in enumerate(words):
        s, e, sc = (float(np.mean([r[i][j] for r in runs])) for j in range(3))
        out.append(dict(word=w, start=round(s, 3), end=round(e, 3), score=round(sc, 3)))
    return out


def _align_once(audio, words, model_name, chunk_s):
    """One forced alignment on the model's own frame grid -> [(start, end, score)] per word."""
    proc, model = load_model(model_name)
    emission = emissions(audio, proc, model, chunk_s)
    frame_s = len(audio) / SR / emission.shape[0]
    vocab = proc.tokenizer.get_vocab()
    blank = proc.tokenizer.pad_token_id
    sep = vocab.get("|")

    tokens, owner = [], []
    for wi, w in enumerate(words):
        chars = [vocab.get(c, vocab.get(c.upper())) for c in re.sub(r"[^\w']", "", w.lower())]
        chars = [c for c in chars if c is not None]
        if not chars:
            continue                                   # e.g. digits: timed from its neighbours below
        if tokens and sep is not None:
            tokens.append(sep)
            owner.append(-1)
        tokens += chars
        owner += [wi] * len(chars)
    if not tokens:
        raise ValueError("nothing in the transcript can be aligned with this model's alphabet")
    if len(tokens) > emission.shape[0]:
        raise ValueError("the transcript is longer than the audio can hold; wrong audio or transcript?")

    spans = {}
    for ti, frame, prob in viterbi(emission, tokens, blank):
        wi = owner[ti]
        if wi < 0:
            continue
        s = spans.setdefault(wi, [frame, frame, []])
        s[0], s[1] = min(s[0], frame), max(s[1], frame)
        s[2].append(prob)
    out = []
    for wi in range(len(words)):
        if wi in spans:
            f0, f1, probs = spans[wi]
            out.append([f0 * frame_s, (f1 + 1) * frame_s, float(np.mean(probs))])
        else:
            out.append([None, None, 0.0])
    # words the model cannot spell (numbers, symbols) sit between their neighbours
    for i, o in enumerate(out):
        if o[0] is None:
            prev_end = next((out[j][1] for j in range(i - 1, -1, -1) if out[j][1] is not None), 0.0)
            next_start = next((out[j][0] for j in range(i + 1, len(out)) if out[j][0] is not None), len(audio) / SR)
            o[0], o[1] = prev_end, max(prev_end, next_start)
    return out
