"""Score a saved LoRA adapter on a cached split, in the same format as the notebooks.

Notebook 2 only evaluates ``data/vi_asr`` test (114 merged segments). Notebook 1
also reports the raw viVoice split (``data/vi_asr_raw``, 1250 clips), which has no
LoRA counterpart — so ``show_results.py`` had a baseline with nothing to compare
against. This produces that counterpart without re-running the notebook.

The raw split is also the better measurement: 1250 utterances instead of 114
tightens the WER confidence interval by ~3.3x, enough to detect improvements the
merged split cannot resolve.

Usage:
    python eval_lora.py                                   # raw viVoice test
    python eval_lora.py --data data/vi_asr --out results/lora
    python eval_lora.py --split val --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import pandas as pd
import torch
from dotenv import load_dotenv

import bench
import data_prep
from vi_norm import wer_cer

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vi_asr_raw",
                    help="cached dataset dir written by data_prep")
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default="checkpoints/vi_lora")
    ap.add_argument("--out", default="results/lora_raw",
                    help="prefix: writes <out>_predictions.csv and <out>_metrics.json")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="lower to 4 if you hit CUDA OOM")
    args = ap.parse_args()

    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    print(f"loading {MODEL_ID} + adapter {args.adapter}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    # Batched generation is only correct with left padding; bench.transcribe_arrays
    # assumes the caller has set this (same contract as both notebooks).
    processor.tokenizer.padding_side = "left"

    rows = list(data_prep.load_splits(args.data)[args.split])
    print(f"{args.data} [{args.split}]: {len(rows)} rows")

    arrays = [data_prep.read_audio(r["audio_path"])[0] for r in rows]
    hyps = bench.transcribe_arrays(model, processor, arrays,
                                   batch_size=args.batch_size)

    df = pd.DataFrame([{
        "audio_file": os.path.basename(r["audio_path"]),
        "audio_path": r["audio_path"],
        "channel": r["channel"], "bucket": r["bucket"],
        "duration": round(r["duration"], 2),
        "ref": r["text"], "hyp": h,
    } for r, h in zip(rows, hyps)])

    overall = wer_cer(df["ref"], df["hyp"])
    by_bucket = {b: wer_cer(g["ref"], g["hyp"]) for b, g in df.groupby("bucket")}

    pathlib.Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.out}_predictions.csv", index=False)
    with open(f"{args.out}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"overall": overall, "by_bucket": by_bucket}, fh,
                  ensure_ascii=False, indent=2)

    print(f"\nOVERALL: WER {overall['wer']:.4f}  CER {overall['cer']:.4f} "
          f"(n={overall['n']})")
    for b, m in by_bucket.items():
        print(f"  {b}: WER={m['wer']:.4f} CER={m['cer']:.4f} (n={m['n']})")
    print(f"\nsaved -> {args.out}_predictions.csv, {args.out}_metrics.json")


if __name__ == "__main__":
    main()
