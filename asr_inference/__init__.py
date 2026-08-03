"""Vietnamese ASR inference: CPU Zipformer RNN-T, or Qwen3-ASR + LoRA on GPU.

``QwenLoRA`` is not imported eagerly -- it pulls in transformers and peft, which
the CPU-only path has no reason to pay for.
"""
from __future__ import annotations

from .results import Result, TimedText
from .zipformer import ZipformerRNNT, load_audio

__all__ = ["ZipformerRNNT", "Result", "TimedText", "load_audio"]
