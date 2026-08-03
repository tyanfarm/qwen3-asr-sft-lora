"""Result types shared by the backends, so both speak the same shape.

``zipformer.py`` (CPU ONNX RNN-T) and ``qwen_lora.py`` (GPU Qwen3-ASR + LoRA)
are entirely different models, but the CLI and any comparison script should not
have to care -- both return a ``Result`` with the same fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimedText:
    start: float
    end: float
    text: str

    def __str__(self) -> str:
        return f"[{hhmmss(self.start)} -> {hhmmss(self.end)}] {self.text}"


@dataclass
class Result:
    text: str
    segments: list[TimedText] = field(default_factory=list)
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def rtf(self) -> float:
        """Real-time factor: < 1 means faster than real time."""
        return self.elapsed_seconds / max(self.audio_seconds, 1e-9)


def hhmmss(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
