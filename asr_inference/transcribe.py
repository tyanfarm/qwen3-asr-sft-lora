"""CLI: transcribe audio files with either ASR backend in this repo.

    python -m asr_inference.transcribe audio.wav
    python -m asr_inference.transcribe lecture.mp3 --format srt -o out.srt
    python -m asr_inference.transcribe *.wav --int8 --threads 8 --lower
    python -m asr_inference.transcribe audio.wav --backend qwen   # LoRA adapter

``--backend zipformer`` (default) is the CPU ONNX RNN-T; ``--backend qwen`` is
Qwen3-ASR-1.7B with one of this repo's LoRA adapters, on GPU. Both share the VAD
chunking and output formats, so the two are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .results import Result, hhmmss
from .zipformer import DEFAULT_MODEL_DIR, ZipformerRNNT


def _srt(result: Result) -> str:
    blocks = []
    for i, seg in enumerate(result.segments, start=1):
        start = hhmmss(seg.start).replace(".", ",")
        end = hhmmss(seg.end).replace(".", ",")
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text}\n")
    return "\n".join(blocks)


def _render(path: Path, result: Result, fmt: str) -> str:
    if fmt == "text":
        return result.text
    if fmt == "segments":
        return "\n".join(str(s) for s in result.segments)
    if fmt == "srt":
        return _srt(result)
    return json.dumps({
        "file": str(path),
        "text": result.text,
        "audio_seconds": round(result.audio_seconds, 3),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "rtf": round(result.rtf, 4),
        "segments": [{"start": round(s.start, 3), "end": round(s.end, 3),
                      "text": s.text} for s in result.segments],
    }, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Vietnamese ASR")
    ap.add_argument("audio", nargs="+", type=Path)
    ap.add_argument("--backend", choices=["zipformer", "qwen"],
                    default="zipformer",
                    help="zipformer: CPU ONNX RNN-T. qwen: Qwen3-ASR + LoRA (GPU)")

    zf = ap.add_argument_group("zipformer backend")
    zf.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    zf.add_argument("--int8", action="store_true",
                    help="quantized encoder: ~2x faster, a little more WER")
    zf.add_argument("--threads", type=int, default=4)

    qw = ap.add_argument_group("qwen backend")
    qw.add_argument("--adapter", type=Path, default=Path("checkpoints/vi_lora"),
                    help="LoRA adapter dir; 'none' for the un-tuned base model")
    qw.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--max-seconds", type=float, default=25.0,
                    help="ceiling per decoded chunk (VAD picks the cut points)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="chunks per encoder pass; raise it for long files")
    ap.add_argument("--no-vad", action="store_true",
                    help="split on energy instead of Silero VAD")
    ap.add_argument("--format", choices=["text", "segments", "srt", "json"],
                    default="text")
    ap.add_argument("--punct", action="store_true",
                    help="restore punctuation and capitalization with "
                         "dragonSwing/xlm-roberta-capu (~1.1 GB, downloaded on "
                         "first use; ~13 s to load, then ~0.015 RTF on top)")
    ap.add_argument("--lower", action="store_true",
                    help="the model emits uppercase; lowercase it for scoring")
    ap.add_argument("-o", "--output", type=Path,
                    help="write here instead of stdout (single input only)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    missing = [p for p in args.audio if not p.exists()]
    if missing:
        print(f"no such file: {missing[0]}", file=sys.stderr)
        return 1
    if args.output and len(args.audio) > 1:
        print("-o takes a single input file", file=sys.stderr)
        return 1
    if args.punct and args.lower:
        print("--punct restores capitalization; --lower undoes it", file=sys.stderr)
        return 1
    if args.punct and args.backend == "qwen":
        print("qwen already emits punctuation; --punct would re-punctuate it",
              file=sys.stderr)
        return 1

    if args.backend == "qwen":
        from .qwen_lora import QwenLoRA, resolve_adapter

        adapter = None if str(args.adapter).lower() == "none" else args.adapter
        if adapter is not None:
            if not adapter.exists():
                print(f"no such adapter: {adapter}", file=sys.stderr)
                return 1
            try:
                adapter = resolve_adapter(adapter)
            except FileNotFoundError as exc:
                print(exc, file=sys.stderr)
                return 1
        if not args.quiet:
            print(f"loading Qwen3-ASR + {adapter or 'no adapter (base model)'}...",
                  file=sys.stderr)
        model = QwenLoRA(adapter, device=args.device)
    else:
        model = ZipformerRNNT(args.model_dir, int8=args.int8,
                              num_threads=args.threads)

    punctuator = None
    if args.punct:
        from .punctuate import Punctuator

        if not args.quiet:
            print("loading punctuation model...", file=sys.stderr)
        punctuator = Punctuator()

    chunks = []
    for path in args.audio:
        def progress(done: int, total: int, _p=path) -> None:
            if not args.quiet and total > 1:
                print(f"\r{_p.name}: chunk {done}/{total}",
                      end="", file=sys.stderr, flush=True)

        result = model.transcribe(path, max_seconds=args.max_seconds,
                                  batch_size=args.batch_size,
                                  use_vad=not args.no_vad,
                                  progress=progress)
        if not args.quiet:
            print(f"\r{path.name}: {result.audio_seconds:.1f}s audio in "
                  f"{result.elapsed_seconds:.1f}s (RTF {result.rtf:.3f}), "
                  f"{len(result.segments)} segment(s)",
                  file=sys.stderr, flush=True)

        if punctuator is not None:
            # Restore per segment, then rebuild the full text from the pieces --
            # punctuating the concatenation instead would give segment texts and
            # `result.text` different wording.
            for seg, text in zip(result.segments,
                                 punctuator.restore_all(
                                     [s.text for s in result.segments])):
                seg.text = text
            result.text = " ".join(s.text for s in result.segments)
        elif args.lower:
            result.text = result.text.lower()
            for seg in result.segments:
                seg.text = seg.text.lower()

        body = _render(path, result, args.format)
        chunks.append(body if len(args.audio) == 1
                      else f"=== {path.name} ===\n{body}")

    out = "\n\n".join(chunks)
    if args.output:
        args.output.write_text(out + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"-> {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
