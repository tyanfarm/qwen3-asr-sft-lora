"""Arbitrary-audio inference with this repo's fine-tuned Qwen3-ASR LoRA adapters.

``eval_lora.py`` scores an adapter on a cached split; this is the other half --
point it at a wav/mp3 and get a transcript, with the same VAD chunking and the
same ``Result`` shape as the Zipformer backend, so the two can be compared with
one command.

Generation itself is delegated to ``bench.transcribe_arrays``, which both
notebooks and ``eval_lora.py`` already use. That keeps one definition of the
generation contract (greedy, left-padded, length-sorted batches) -- a second
copy here would drift and quietly make CLI transcripts differ from scored ones.

Unlike the Zipformer, Qwen3-ASR emits **cased, punctuated** text natively, so
``--punct`` is neither needed nor offered for this backend.

This is a 1.7B model: on GPU it runs comfortably, on CPU it is slow enough that
you would rather use the Zipformer. The device defaults to CUDA when available.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .results import Result, TimedText
from .vad import SileroVAD, segment_audio
from .zipformer import TARGET_SR, load_audio

BASE_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
DEFAULT_ADAPTER = Path("checkpoints/vi_lora")


def resolve_adapter(path: str | Path) -> Path:
    """Point at a run directory, get the adapter it should actually load.

    A finished run has ``save_pretrained`` output at the top level, and that is
    what to use -- with ``load_best_model_at_end`` it is the *best* checkpoint,
    which is not always the last one. A run that is still going (or was killed)
    has only ``checkpoint-N/`` subdirectories, and there the highest N is the
    best available answer.

    Passing a ``checkpoint-N`` directory directly still works: it has its own
    ``adapter_config.json``, so the first branch takes it verbatim.
    """
    path = Path(path)
    if (path / "adapter_config.json").exists():
        return path

    checkpoints = [(int(p.name.split("-")[-1]), p)
                   for p in path.glob("checkpoint-*")
                   if p.name.split("-")[-1].isdigit()
                   and (p / "adapter_config.json").exists()]
    if not checkpoints:
        raise FileNotFoundError(
            f"no adapter in {path} -- expected adapter_config.json or a "
            f"checkpoint-*/ subdirectory containing one")
    return max(checkpoints)[1]


class QwenLoRA:
    """Qwen3-ASR with a PEFT adapter, wrapped in the shared Result interface."""

    def __init__(self, adapter: str | Path | None = DEFAULT_ADAPTER, *,
                 base_model: str = BASE_MODEL_ID, device: str | None = None,
                 language: str = "Vietnamese") -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language

        self.processor = AutoProcessor.from_pretrained(base_model)
        model = AutoModelForMultimodalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=self.device,
        )
        if adapter is not None:
            from peft import PeftModel

            adapter = resolve_adapter(adapter)
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        self.model = model
        # Batched generation is only correct with left padding -- the same
        # contract bench.transcribe_arrays assumes of every caller.
        self.processor.tokenizer.padding_side = "left"

        self.adapter = str(adapter) if adapter is not None else None
        self._vad: SileroVAD | None = None

    def transcribe(self, audio: str | Path | np.ndarray, *,
                   sample_rate: int = TARGET_SR,
                   max_seconds: float = 25.0,
                   batch_size: int = 4,
                   use_vad: bool = True,
                   progress=None) -> Result:
        """Transcribe a file path or a float array in [-1, 1], of any length.

        ``max_seconds`` defaults to 25 because the adapter was trained on 5-30 s
        segments; feeding it a much longer span is out of distribution even
        though the base model would accept it.
        """
        import bench  # repo-root module; see the module docstring

        if isinstance(audio, (str, Path)):
            samples, sample_rate = load_audio(audio), TARGET_SR
        else:
            samples = np.ascontiguousarray(audio, dtype=np.float32)

        started = time.perf_counter()
        spans = segment_audio(samples, sample_rate, max_seconds=max_seconds,
                              vad=self._vad_or_load() if use_vad else None,
                              use_vad=use_vad)
        texts: list[str] = []
        if spans:
            texts = bench.transcribe_arrays(
                self.model, self.processor,
                [s.slice(samples, sample_rate) for s in spans],
                batch_size=batch_size, language=self.language,
            )
        if progress is not None:
            progress(len(spans), len(spans))
        elapsed = time.perf_counter() - started

        timed = [TimedText(s.start, s.end, (t or "").strip())
                 for s, t in zip(spans, texts) if (t or "").strip()]
        return Result(
            text=" ".join(t.text for t in timed),
            segments=timed,
            audio_seconds=len(samples) / sample_rate,
            elapsed_seconds=elapsed,
        )

    def _vad_or_load(self) -> SileroVAD | None:
        if self._vad is None:
            try:
                self._vad = SileroVAD()
            except ImportError:
                return None
        return self._vad
