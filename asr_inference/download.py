"""Fetch the Zipformer RNN-T weights from the Hub into ``asr_inference/models``.

Kept separate from ``zipformer.py`` so the recognizer never reaches for the
network at import time -- benchmarks should fail loudly on a missing file, not
silently start a 90 MB download mid-run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "hynt/Zipformer-30M-RNNT-6000h"
DEFAULT_DIR = Path(__file__).resolve().parent / "models"

# ``config.json`` is misleadingly named: it is icefall's tokens.txt (one
# "<piece> <id>" per line, 2000 BPE pieces). ``bpe.model`` is only needed if you
# want to re-tokenize text yourself; decoding does not use it.
COMMON = ["config.json", "bpe.model"]
FP32 = [f"{p}-epoch-20-avg-10.onnx" for p in ("encoder", "decoder", "joiner")]
INT8 = [f"{p}-epoch-20-avg-10.int8.onnx" for p in ("encoder", "decoder", "joiner")]


def download(dest: Path = DEFAULT_DIR, *, int8: bool = True, fp32: bool = True) -> Path:
    from huggingface_hub import hf_hub_download

    files = COMMON + (FP32 if fp32 else []) + (INT8 if int8 else [])
    dest.mkdir(parents=True, exist_ok=True)
    for name in files:
        print(f"  {name}")
        hf_hub_download(REPO_ID, name, local_dir=str(dest))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Download {REPO_ID}")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--only", choices=["fp32", "int8", "both"], default="both")
    ap.add_argument("--punct", action="store_true",
                    help="also fetch the ~1.1 GB punctuation/capitalization model")
    args = ap.parse_args()

    out = download(args.dest,
                   int8=args.only in ("int8", "both"),
                   fp32=args.only in ("fp32", "both"))
    print(f"-> {out}")

    if args.punct:
        from .punctuate import download as download_capu

        print("punctuation model...")
        print(f"-> {download_capu()}")


if __name__ == "__main__":
    main()
