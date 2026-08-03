"""Voice-activity segmentation, so an hour of audio can reach a 30 s encoder.

The Zipformer here is **offline**: it consumes a whole utterance at once and its
attention cost grows with the square of the input length. Feeding it a 1-hour
file is not slow-but-fine, it is tens of GB of activations. So long audio has to
be cut, and *where* it is cut decides the WER -- a cut through the middle of a
word costs that word twice (truncated on the left chunk, truncated on the
right).

Silero VAD is the default because it is a 2 MB ONNX model that runs at ~1% of
real time on CPU, and it tracks speech rather than loudness, so it survives
background noise that a dB-threshold splitter would happily transcribe as
speech. ``split_on_energy`` stays as a dependency-free fallback.

Two knobs matter and pull against each other:

* ``max_seconds`` -- the hard ceiling per chunk. Larger means fewer boundaries
  (better context, better WER) but more encoder memory. 20-30 s is the sweet
  spot for a 30M zipformer on CPU.
* ``speech_pad`` -- how much audio to keep either side of a detected segment.
  VAD trims onsets and offsets slightly, and a clipped first phoneme is a
  reliable way to lose the first word.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_SR = 16_000


@dataclass(frozen=True)
class Segment:
    """A span of the source audio, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def slice(self, samples: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
        lo = max(0, int(self.start * sample_rate))
        hi = min(len(samples), int(self.end * sample_rate))
        return samples[lo:hi]


class SileroVAD:
    """Thin wrapper over the ``silero-vad`` package, held so the model loads once."""

    def __init__(self, *, threshold: float = 0.5, min_speech_ms: int = 250,
                 min_silence_ms: int = 300, speech_pad_ms: int = 200,
                 onnx: bool = True) -> None:
        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=onnx)
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

    def segments(self, samples: np.ndarray, sample_rate: int = TARGET_SR, *,
                 max_seconds: float = 30.0,
                 progress=None) -> list[Segment]:
        """Detected speech spans, none longer than ``max_seconds``.

        ``max_speech_duration_s`` makes Silero itself break an over-long run at
        its quietest interior point, which is a far better cut than an
        arbitrary one -- so we never have to hard-cut speech here.
        """
        import torch
        from silero_vad import get_speech_timestamps

        wav = torch.from_numpy(np.asarray(samples, dtype=np.float32))
        stamps = get_speech_timestamps(
            wav,
            self.model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            # Leave the padding to merge_segments: Silero's speech_pad_ms is
            # applied per detection, and we merge detections afterwards.
            speech_pad_ms=0,
            max_speech_duration_s=max_seconds,
            return_seconds=True,
            progress_tracking_callback=progress,
        )
        return [Segment(float(s["start"]), float(s["end"])) for s in stamps]


def merge_segments(segments: list[Segment], *, max_seconds: float = 30.0,
                   max_gap: float = 0.8, pad: float = 0.2,
                   total_seconds: float | None = None) -> list[Segment]:
    """Glue neighbouring speech spans into as-long-as-allowed decoding chunks.

    VAD emits one span per breath group; decoding those individually would both
    starve the encoder of context and pay the per-call overhead hundreds of
    times over an hour. Merging while the accumulated span fits in
    ``max_seconds`` and the pause between spans is under ``max_gap`` gives
    chunks that end at real pauses.
    """
    if not segments:
        return []

    merged: list[Segment] = []
    start, end = segments[0].start, segments[0].end
    for seg in segments[1:]:
        fits = seg.end - start <= max_seconds
        contiguous = seg.start - end <= max_gap
        if fits and contiguous:
            end = seg.end
        else:
            merged.append(Segment(start, end))
            start, end = seg.start, seg.end
    merged.append(Segment(start, end))

    if pad <= 0:
        return merged
    limit = total_seconds if total_seconds is not None else merged[-1].end + pad
    return [Segment(max(0.0, s.start - pad), min(limit, s.end + pad)) for s in merged]


def split_on_energy(samples: np.ndarray, *, max_seconds: float = 30.0,
                    sample_rate: int = TARGET_SR,
                    top_db: float = 35.0) -> list[Segment]:
    """Dependency-free fallback: split at dB minima instead of at speech.

    Used when ``silero-vad`` is not installed. Noticeably worse on noisy audio,
    where a fan or a road is "loud enough" to look like speech and the chunks
    end up hard-cut at the ceiling.
    """
    import librosa

    if len(samples) <= max_seconds * sample_rate:
        return [Segment(0.0, len(samples) / sample_rate)]

    intervals = librosa.effects.split(samples, top_db=top_db)
    if len(intervals) == 0:
        intervals = np.array([[0, len(samples)]])
    spans = [Segment(int(lo) / sample_rate, int(hi) / sample_rate)
             for lo, hi in intervals]

    # librosa gives no length guarantee, so anything still over the ceiling
    # after merging has to be hard-cut -- the case VAD avoids.
    out: list[Segment] = []
    for seg in merge_segments(spans, max_seconds=max_seconds, pad=0.0):
        if seg.duration <= max_seconds:
            out.append(seg)
            continue
        n = int(np.ceil(seg.duration / max_seconds))
        step = seg.duration / n
        out += [Segment(seg.start + i * step, seg.start + (i + 1) * step)
                for i in range(n)]
    return out


def segment_audio(samples: np.ndarray, sample_rate: int = TARGET_SR, *,
                  max_seconds: float = 30.0, max_gap: float = 0.8,
                  pad: float = 0.2, vad: SileroVAD | None = None,
                  use_vad: bool = True, progress=None) -> list[Segment]:
    """Cut audio into decodable chunks: Silero VAD if available, energy if not."""
    total = len(samples) / sample_rate
    if total <= max_seconds:
        return [Segment(0.0, total)]

    if use_vad:
        try:
            vad = vad or SileroVAD()
            spans = vad.segments(samples, sample_rate,
                                 max_seconds=max_seconds, progress=progress)
            if spans:
                return merge_segments(spans, max_seconds=max_seconds,
                                      max_gap=max_gap, pad=pad,
                                      total_seconds=total)
            return []  # genuinely no speech; do not fall through to energy
        except ImportError:
            pass
    return split_on_energy(samples, max_seconds=max_seconds,
                           sample_rate=sample_rate)
