"""CPU inference for the Vietnamese Zipformer-30M RNN-T (hynt, VLSP 2025).

The Hub repo ships the sherpa-onnx export of an icefall ``zipformer2``
transducer: three ONNX graphs (encoder / decoder / joiner) plus a 2000-piece
BPE vocabulary. Rather than depend on k2 or a sherpa-onnx wheel -- neither of
which is in this project's requirements, and both of which are awkward to build
-- we drive the graphs directly with onnxruntime. That is only ~150 lines,
because an offline transducer carries no state between calls:

    fbank -> encoder -> [greedy loop over frames: joiner(enc[t], dec)] -> text

Feature contract (must match how icefall trained the model, or the output is
fluent-sounding garbage):

* 16 kHz mono, samples in [-1, 1] -- **not** int16 range. lhotse feeds
  torchaudio's kaldi-compliance fbank normalized floats, and sherpa-onnx does
  the same, so we do too.
* 80 Mel bins, 25 ms / 10 ms, ``low_freq=20``, ``high_freq=-400`` (i.e. 7600 Hz),
  ``dither=0``, ``snip_edges=False``. These are lhotse's ``FbankConfig``
  defaults, which is what icefall used.

Long audio is handled by ``vad.py``: the encoder is non-streaming, so an hour of
audio is segmented on speech activity and decoded chunk by chunk, with
timestamps preserved.

The model emits **uppercase, unpunctuated** Vietnamese ("XIN CHÀO"), because
that is how its BPE vocabulary was cased. Lowercase it before scoring against
this repo's references -- ``vi_norm.normalize`` already does.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .results import Result, TimedText, hhmmss  # noqa: F401  (re-exported)
from .vad import Segment, SileroVAD, segment_audio

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
TARGET_SR = 16_000

# icefall's BPE marks a word boundary with U+2581, not a space.
_WORD_BOUNDARY = "▁"


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read any soundfile-supported file as mono float32 in [-1, 1] at 16 kHz."""
    import soundfile as sf

    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)  # downmix; the model is mono-only
    if sr != target_sr:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _fbank(samples: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
    """80-dim log-Mel fbank, (T, 80) float32. See the module docstring."""
    import torch
    import torchaudio.compliance.kaldi as kaldi

    wave = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
    feats = kaldi.fbank(
        wave,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        low_freq=20.0,
        high_freq=-400.0,
        sample_frequency=float(sample_rate),
        snip_edges=False,
        window_type="povey",
        use_energy=False,
    )
    return feats.numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# recognizer
# --------------------------------------------------------------------------- #
class ZipformerRNNT:
    """Offline (non-streaming) transducer decoder running on CPU."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, *,
                 int8: bool = False, num_threads: int = 4,
                 epoch_tag: str = "epoch-20-avg-10") -> None:
        import onnxruntime as ort

        # This is the CPU path, so ort's provider chatter is never interesting.
        # (Its "GPU device discovery failed" line on WSL fires during import,
        # before any Python call can mute it -- that one you just live with.)
        ort.set_default_logger_severity(3)

        model_dir = Path(model_dir)
        suffix = ".int8.onnx" if int8 else ".onnx"

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # onnxruntime is chatty about missing GPUs

        def session(part: str):
            path = model_dir / f"{part}-{epoch_tag}{suffix}"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found -- run `python -m asr_inference.download`")
            return ort.InferenceSession(str(path), opts,
                                        providers=["CPUExecutionProvider"])

        self.encoder = session("encoder")
        self.decoder = session("decoder")
        self.joiner = session("joiner")

        meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(meta["context_size"])
        self.vocab_size = int(meta["vocab_size"])
        self.blank_id = 0
        self.tokens = _read_tokens(model_dir / "config.json")
        if len(self.tokens) != self.vocab_size:
            raise ValueError(f"tokens={len(self.tokens)} vs vocab={self.vocab_size}")

        self._vad: SileroVAD | None = None

    # -- public API --------------------------------------------------------- #
    def transcribe(self, audio: str | Path | np.ndarray, *,
                   sample_rate: int = TARGET_SR,
                   max_seconds: float = 25.0,
                   batch_size: int = 4,
                   use_vad: bool = True,
                   progress=None) -> Result:
        """Transcribe a file path or a float array in [-1, 1], of any length.

        Audio longer than ``max_seconds`` is segmented on voice activity and
        decoded chunk by chunk; ``Result.segments`` carries the timestamps, and
        ``Result.text`` the whole transcript. ``progress`` is called with
        (chunks_done, chunks_total) -- worth wiring up for hour-long files.
        """
        if isinstance(audio, (str, Path)):
            samples, sample_rate = load_audio(audio), TARGET_SR
        else:
            samples = np.ascontiguousarray(audio, dtype=np.float32)

        started = time.perf_counter()
        spans = segment_audio(samples, sample_rate, max_seconds=max_seconds,
                              vad=self._vad_or_load() if use_vad else None,
                              use_vad=use_vad)
        texts = self._decode_spans(samples, sample_rate, spans,
                                   batch_size=batch_size, progress=progress)
        elapsed = time.perf_counter() - started

        timed = [TimedText(s.start, s.end, t)
                 for s, t in zip(spans, texts) if t]
        return Result(
            text=" ".join(t.text for t in timed),
            segments=timed,
            audio_seconds=len(samples) / sample_rate,
            elapsed_seconds=elapsed,
        )

    def transcribe_batch(self, items, *, sample_rate: int = TARGET_SR) -> list[str]:
        """Transcribe several *short* utterances in one encoder pass.

        For benchmark corpora, where every clip already fits the encoder. No VAD
        and no length sorting -- padding is to the longest item, so hand this
        similarly-sized clips or the encoder burns its work on zeros.
        """
        feats = []
        for item in items:
            if isinstance(item, (str, Path)):
                feats.append(_fbank(load_audio(item), TARGET_SR))
            else:
                feats.append(_fbank(np.ascontiguousarray(item, dtype=np.float32),
                                    sample_rate))
        return [_detokenize(h) for h in self._decode(feats)]

    # -- internals ---------------------------------------------------------- #
    def _vad_or_load(self) -> SileroVAD | None:
        if self._vad is None:
            try:
                self._vad = SileroVAD()
            except ImportError:
                return None
        return self._vad

    def _decode_spans(self, samples: np.ndarray, sample_rate: int,
                      spans: list[Segment], *, batch_size: int,
                      progress=None) -> list[str]:
        """Decode VAD chunks, batching similar lengths together.

        Batches are formed over *duration-sorted* indices: a batch pads to its
        longest member, so pairing a 3 s chunk with a 25 s one would waste ~90%
        of the encoder pass on that row. Results are scattered back into the
        original chunk order at the end.
        """
        if not spans:
            return []

        order = sorted(range(len(spans)), key=lambda i: spans[i].duration)
        texts: list[str] = [""] * len(spans)
        done = 0
        for start in range(0, len(order), max(1, batch_size)):
            group = order[start:start + max(1, batch_size)]
            feats = [_fbank(spans[i].slice(samples, sample_rate), sample_rate)
                     for i in group]
            for i, hyp in zip(group, self._decode(feats)):
                texts[i] = _detokenize(hyp)
            done += len(group)
            if progress is not None:
                progress(done, len(spans))
        return texts

    def _decode(self, feats: list[np.ndarray]) -> list[list[str]]:
        if not feats:
            return []
        lens = np.array([f.shape[0] for f in feats], dtype=np.int64)
        padded = np.zeros((len(feats), int(lens.max()), 80), dtype=np.float32)
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f

        enc_out, enc_lens = self.encoder.run(
            ["encoder_out", "encoder_out_lens"], {"x": padded, "x_lens": lens})
        return self._greedy_search(enc_out, enc_lens)

    def _greedy_search(self, enc_out: np.ndarray,
                       enc_lens: np.ndarray) -> list[list[str]]:
        """Batched RNN-T greedy search (icefall's ``greedy_search_batch``).

        At most one symbol is emitted per frame, so the loop is bounded by T and
        cannot run away. The decoder is re-run only for the rows that emitted,
        which is what keeps the per-frame cost close to the joiner alone.
        """
        n, total_frames, _ = enc_out.shape
        hyps = [[self.blank_id] * self.context_size for _ in range(n)]

        dec_in = np.array([h[-self.context_size:] for h in hyps], dtype=np.int64)
        dec_out = self.decoder.run(["decoder_out"], {"y": dec_in})[0]

        for t in range(total_frames):
            active = [i for i in range(n) if t < enc_lens[i]]
            if not active:
                break
            rows = np.asarray(active)
            logits = self.joiner.run(
                ["logit"],
                {"encoder_out": enc_out[rows, t], "decoder_out": dec_out[rows]},
            )[0]

            emitted = []
            for slot, i in enumerate(active):
                token = int(logits[slot].argmax())
                if token != self.blank_id:
                    hyps[i].append(token)
                    emitted.append(i)
            if emitted:
                sub = np.array([hyps[i][-self.context_size:] for i in emitted],
                               dtype=np.int64)
                dec_out[np.asarray(emitted)] = self.decoder.run(
                    ["decoder_out"], {"y": sub})[0]

        return [[self.tokens[t] for t in h[self.context_size:]] for h in hyps]


# --------------------------------------------------------------------------- #
# tokens / formatting
# --------------------------------------------------------------------------- #
def _read_tokens(path: Path) -> list[str]:
    """Read icefall's ``tokens.txt`` (shipped here under the name config.json)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m asr_inference.download`")
    table: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # rsplit: the piece itself may contain a space (e.g. the literal " ").
        piece, index = line.rsplit(" ", maxsplit=1)
        table[int(index)] = piece
    return [table[i] for i in range(len(table))]


def _detokenize(tokens: list[str]) -> str:
    return "".join(tokens).replace(_WORD_BOUNDARY, " ").strip()
