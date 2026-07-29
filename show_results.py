"""Pretty-print the evaluation results written to ``results/``.

Usage:
    python show_results.py                      # LoRA on raw viVoice test, vs baseline
    python show_results.py --model baseline     # rank the base model's rows instead
    python show_results.py --set merged         # the 114-clip merged split
    python show_results.py --set vivos --n 10   # a section-9 benchmark
    python show_results.py --csv path/to.csv    # anything else

Prints the metric summary (overall + per duration bucket), a before/after
comparison, and the best/worst-WER predictions so you can eyeball where the
model struggles. When both systems have predictions for the same rows, each
example also shows the other system's hypothesis, so a bad row reads
immediately as "both got it wrong" or "the fine-tune broke this one".
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from vi_norm import normalize_vi  # the one definition, shared with the notebooks

# Each eval set names both systems' artefacts. Pairing them here is what keeps
# the before/after honest: an earlier version of this script compared
# baseline_raw (1250 raw clips) against lora (114 merged segments) — different
# test sets, so the reported delta was measuring the split, not the model.
SETS: dict[str, dict] = {
    # Default: the largest held-out viVoice set, so row-level examples and the
    # WER delta both come from the same 1250 clips.
    "raw": {
        "label": "viVoice test — raw clips (1250, speaker-disjoint, unseen)",
        "baseline": ("results/baseline_raw_predictions.csv",
                     "results/baseline_raw_metrics.json"),
        "lora": ("results/lora_raw_predictions.csv",
                 "results/lora_raw_metrics.json"),
    },
    "merged": {
        "label": "viVoice test — merged segments (114, the notebook-2 number)",
        "baseline": ("results/baseline_predictions.csv",
                     "results/baseline_metrics.json"),
        "lora": ("results/lora_predictions.csv", "results/lora_metrics.json"),
    },
}
# The section-9 benchmarks share a shape: one metrics file holds every set.
for _name, _label in [
    ("vivos", "VIVOS test — in-domain after fine-tuning"),
    ("cmv_vi", "Common Voice 17 vi test — in-domain after fine-tuning"),
    ("vlsp2020_100h", "VLSP 2020 held-out slice — in-domain after fine-tuning"),
]:
    SETS[_name] = {
        "label": _label,
        "baseline": (f"results/bench_{_name}_predictions.csv",
                     ("results/bench_metrics.json", _name)),
        "lora": (f"results/lora_bench_{_name}_predictions.csv",
                 ("results/lora_bench_metrics.json", _name)),
    }


def row_wer(ref: str, hyp: str, expand_numbers: bool = True) -> float:
    import jiwer

    r = normalize_vi(ref, expand_numbers=expand_numbers)
    h = normalize_vi(hyp, expand_numbers=expand_numbers)
    if not r:
        return float("nan")
    return jiwer.wer(r, h)


def load_metrics(spec) -> dict | None:
    """Read a metrics file, unwrapping the ``(path, key)`` benchmark form.

    Benchmark files are flat ``{name: {wer, cer, n}}`` while the viVoice files
    nest under ``overall``/``by_bucket``. Both come back in the nested shape.
    """
    if spec is None:
        return None
    path, key = spec if isinstance(spec, tuple) else (spec, None)
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    if key is None:
        return data
    if key not in data:
        return None
    return {"overall": data[key], "by_bucket": {}}


def _fmt_metrics(name: str, m: dict) -> None:
    o = m["overall"]
    print(f"\n{name}")
    print(f"  overall : WER {o['wer']:.4f}  CER {o['cer']:.4f}  (n={o['n']})")
    if "wer_legacy" in o:
        print(f"  legacy  : WER {o['wer_legacy']:.4f}  CER {o['cer_legacy']:.4f}"
              "   (no number expansion)")
    for bucket, mm in m.get("by_bucket", {}).items():
        print(f"  {bucket:<7}: WER {mm['wer']:.4f}  CER {mm['cer']:.4f}  (n={mm['n']})")


def align(df: pd.DataFrame, other_csv: str | None) -> pd.Series | None:
    """Return the other system's hypotheses, row-aligned to ``df``, else None.

    Returning None rather than guessing matters: a misaligned hypothesis column
    would silently print one utterance's output under another's reference.
    """
    if not other_csv or not os.path.exists(other_csv):
        return None
    other = pd.read_csv(other_csv, keep_default_na=False)
    if "audio_file" in df.columns and "audio_file" in other.columns:
        merged = df[["audio_file"]].merge(other[["audio_file", "hyp"]],
                                          on="audio_file", how="left")
        if len(merged) == len(df) and merged["hyp"].notna().all():
            return merged["hyp"]
        return None
    # Benchmark CSVs have no key column: fall back to position, but only when
    # the references agree row for row.
    if len(other) == len(df) and (other["ref"].astype(str).values
                                  == df["ref"].astype(str).values).all():
        return other["hyp"].reset_index(drop=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", default="raw", choices=sorted(SETS),
                    help="which evaluation set to report (default: raw)")
    ap.add_argument("--model", default="lora", choices=("lora", "baseline"),
                    help="whose predictions to rank (default: lora)")
    ap.add_argument("--csv", default=None,
                    help="predictions CSV to sample from (overrides --set/--model)")
    ap.add_argument("--n", type=int, default=5, help="number of example rows to show")
    args = ap.parse_args()

    spec = SETS[args.set_name]
    print(f"=== {spec['label']} ===")

    # ---- metrics summary ---------------------------------------------------- #
    base = load_metrics(spec["baseline"][1])
    lora = load_metrics(spec["lora"][1])
    if base:
        _fmt_metrics("BASELINE", base)
    if lora:
        _fmt_metrics("LoRA", lora)
    if not base and not lora:
        print("\n(no metrics for this set yet — run the notebooks, or eval_lora.py)")

    if base and lora:
        if base["overall"]["n"] != lora["overall"]["n"]:
            print(f"\nWARNING: baseline scored {base['overall']['n']} rows but LoRA "
                  f"scored {lora['overall']['n']} — different test sets, so the "
                  "comparison below is not valid.")
        print("\nBEFORE / AFTER")
        comp = pd.DataFrame({
            "metric": ["WER", "CER"],
            "baseline": [base["overall"]["wer"], base["overall"]["cer"]],
            "lora": [lora["overall"]["wer"], lora["overall"]["cer"]],
        })
        comp["abs_delta"] = comp["lora"] - comp["baseline"]
        comp["rel_%"] = 100 * comp["abs_delta"] / comp["baseline"]
        print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- predictions -------------------------------------------------------- #
    csv = args.csv or spec[args.model][0]
    other_name = "baseline" if args.model == "lora" else "lora"
    other_csv = None if args.csv else spec[other_name][0]

    if not os.path.exists(csv):
        print(f"\n(no predictions CSV at {csv})")
        if args.model == "lora" and args.set_name == "raw":
            print("    build it with:  python eval_lora.py")
        return

    df = pd.read_csv(csv, keep_default_na=False)  # keep empty hyp as "" not NaN
    print(f"\nPREDICTIONS [{args.model}]: {len(df)} rows from {csv}")
    if "bucket" in df.columns:
        print("buckets:", df["bucket"].value_counts().to_dict())

    df["wer"] = [row_wer(r, h) for r, h in zip(df["ref"], df["hyp"])]
    df["wer_legacy"] = [row_wer(r, h, expand_numbers=False)
                        for r, h in zip(df["ref"], df["hyp"])]

    other_hyp = align(df, other_csv)
    if other_hyp is not None:
        df[f"hyp_{other_name}"] = other_hyp
        df[f"wer_{other_name}"] = [row_wer(r, h)
                                   for r, h in zip(df["ref"], other_hyp)]
    elif other_csv and os.path.exists(other_csv):
        print(f"(could not row-align {other_csv}; showing {args.model} only)")

    # How much of the WER was just digits-vs-spoken formatting?
    has_digits = df["ref"].astype(str).str.contains(r"\d")
    print(f"\nNUMBER NORMALIZATION ({has_digits.sum()} of {len(df)} refs "
          f"contain digits)")
    print(f"  mean row WER  without number expansion : {df['wer_legacy'].mean():.4f}")
    print(f"  mean row WER  with number expansion    : {df['wer'].mean():.4f}")
    if has_digits.any():
        sub = df[has_digits]
        print(f"  ...on digit rows only : {sub['wer_legacy'].mean():.4f}"
              f" -> {sub['wer'].mean():.4f}")

    file_col = "audio_file" if "audio_file" in df.columns else (
        "audio_path" if "audio_path" in df.columns else None)

    def _show(title: str, sub: pd.DataFrame) -> None:
        print(f"\n--- {title} ---")
        for _, r in sub.iterrows():
            fname = f" | {r[file_col]}" if file_col else ""
            bucket = f"{r['bucket']} | " if "bucket" in df.columns else ""
            print(f"[{bucket}{r['duration']:.1f}s | WER {r['wer']:.3f}"
                  f" (legacy {r['wer_legacy']:.3f}){fname}]")
            print(f"  REF      : {r['ref']}")
            print(f"  {args.model.upper():<8} : {r['hyp']}")
            if other_hyp is not None:
                delta = r["wer"] - r[f"wer_{other_name}"]
                verdict = ("worse" if delta > 1e-9
                           else "better" if delta < -1e-9 else "same")
                print(f"  {other_name.upper():<8} : {r[f'hyp_{other_name}']}")
                print(f"  {'':8}   {r[f'wer_{other_name}']:.3f} -> {r['wer']:.3f}"
                      f"  ({verdict} {delta:+.3f})")

    _show(f"worst {args.n} by WER [{args.model}]", df.nlargest(args.n, "wer"))
    _show(f"best {args.n} by WER [{args.model}]", df.nsmallest(args.n, "wer"))


if __name__ == "__main__":
    main()
