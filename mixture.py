"""Multi-source Vietnamese training mixture for the LoRA fine-tune.

``data_prep.py`` caches the viVoice corpus this project started from, and
``bench.py`` streams the published *test* sets. This module assembles the
*training* mixture: viVoice plus the training splits of VIVOS, Common Voice 17
Vietnamese and VLSP 2020, capped so no single source dominates.

Records use exactly the schema ``data_prep`` writes — ``audio_path``, ``text``,
``duration``, ``channel``, ``bucket`` (plus ``source``) — so ``load_splits``,
``read_audio`` and the notebook's collator all work unchanged.

One target style
----------------
Sources do not agree on transcript conventions, and the model fits whatever it
is shown. VIVOS is ALL CAPS; the rest are ordinary case. ``normalize_train_text``
folds that difference on ingest — see its docstring for why the first run lost
4.5 WER points on VIVOS without it. Punctuation still differs (viVoice punctuates,
VIVOS and VLSP do not), which is a milder conflict: ``vi_norm`` strips punctuation
before scoring *and* those tokens are in-distribution for the tokenizer, so it
costs capacity but not token-space coherence. Left alone deliberately.

Keeping the eval sets clean
---------------------------
VIVOS and Common Voice have official test splits, so we train on their ``train``
splits and never touch the test rows. Their benchmark numbers stop being
*zero-shot* after this, which the report must say.

VLSP 2020 has no splits at all — only ``train``. Rows are therefore assigned by
hashing the **transcript**: identical sentences always land in the same bucket,
so a sentence cannot appear in both training and evaluation. The held-out slice
replaces the earlier VLSP benchmark, which was sampled from the same rows this
mixture now trains on.
"""
from __future__ import annotations

import hashlib
import os
import unicodedata
from dataclasses import dataclass

TARGET_SR = 16_000

# Bumped whenever a change alters the *content* of the cached mixture, so a
# stale data/vi_mix/ is rebuilt instead of silently reused.
#   v2 - transcripts are case-normalized on ingest (see normalize_train_text).
MIX_VERSION = 2

# Validation runs every epoch, so keep the optional official val splits small.
VAL_HOURS_CAP = 0.5

# train / val / test. Small eval slices: VLSP is big, and 5% of ~11h is plenty.
HASH_RATIOS = (0.90, 0.05, 0.05)
_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class MixSource:
    """One contributor to the training mixture."""

    repo: str | None = None          # HF dataset id; None -> local cache
    cached_dir: str | None = None    # local dir written by data_prep
    config: str | None = None
    train_split: str | None = None
    val_split: str | None = None     # official val split, if the dataset has one
    text_column: str | None = None
    hours: float | None = None       # cap; None -> take everything
    hash_split: bool = False         # no official splits -> bucket by transcript


MIX: dict[str, MixSource] = {
    # Already on disk with channel-disjoint splits; reuse it rather than
    # re-streaming ~8h of viVoice.
    "vivoice": MixSource(cached_dir="data/vi_asr"),
    "vivos": MixSource(
        repo="htdung167/vivos-preprocessed-v2",
        train_split="train",
        text_column="original_sentence",
        hours=15.0,
    ),
    "cmv_vi": MixSource(
        repo="fixie-ai/common_voice_17_0",
        config="vi",
        train_split="train",
        val_split="validation",
        text_column="sentence",
        hours=3.0,
    ),
    # Capped hard: the full corpus is ~100h and would be 80% of the mixture.
    "vlsp2020_100h": MixSource(
        repo="doof-ferb/vlsp2020_vinai_100h",
        train_split="train",
        text_column="transcription",
        hours=11.0,
        hash_split=True,
    ),
}


# --- pure helpers ----------------------------------------------------------- #
def normalize_train_text(text: str) -> str:
    """Put every source's transcripts into one casing style.

    VIVOS ships its transcripts in ALL CAPS; viVoice, Common Voice and VLSP are
    ordinary case. Left alone that is 59% of the mixture (11,660 of 19,881
    rows), and the model learns to condition casing on *acoustic domain* —
    after the first fine-tune 100% of VIVOS test hypotheses came back uppercase
    while every other benchmark stayed lowercase.

    That is not cosmetic. ``vi_norm`` lowercases before scoring, so casing is
    free at the metric level — yet VIVOS test WER still went 7.19% -> 11.64%.
    The reason is tokenization: uppercase Vietnamese has no whole-word tokens
    in the Qwen vocabulary, so "VÀ NẾU BẠN" fragments to
    ``['V','À','_N','Ế','U','_B','Ạ','N']`` where "và nếu bạn" is three tokens
    (2.04x more tokens overall). Generation then runs at roughly character
    granularity in a token space the base model has almost no language-model
    prior over — and that prior is what resolves acoustically ambiguous audio.
    Errors that appear are phonetic, not typographic: "trót" -> "TÓT",
    "rối loạn co bóp" -> "ĐÓI LỌN CÓ BỚP".

    Only all-caps rows are folded. Lowercasing everything would strip the
    proper-noun capitals that viVoice and Common Voice legitimately carry.
    """
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return text.lower()
    return text


def hash_bucket(text: str, ratios: tuple[float, float, float] = HASH_RATIOS) -> str:
    """Deterministically assign a transcript to train/val/test.

    Hashing the text (not the row position) keeps duplicate sentences out of two
    splits at once, and survives any change in dataset iteration order.
    """
    key = unicodedata.normalize("NFC", text).strip().lower().encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    point = int(digest, 16) / float(1 << 128)
    cumulative = 0.0
    for name, share in zip(_SPLITS, ratios):
        cumulative += share
        if point < cumulative:
            return name
    return _SPLITS[-1]


def take_until_hours(rows, hours: float | None):
    """Yield rows until ``hours`` of audio has been emitted (lazily).

    ``hours=None`` passes everything through.
    """
    if hours is None:
        yield from rows
        return
    budget = hours * 3600.0
    total = 0.0
    for row in rows:
        if total >= budget:
            return
        total += row["duration"]
        yield row


def summarize(records):
    """Per-source row count and hours, for printing the mixture composition."""
    import pandas as pd

    if not records:
        return pd.DataFrame(columns=["source", "n", "hours"])
    df = pd.DataFrame([{"source": r["source"], "duration": r["duration"]}
                       for r in records])
    out = (df.groupby("source")
             .agg(n=("duration", "size"), hours=("duration", "sum"))
             .reset_index())
    out["hours"] = (out["hours"] / 3600.0).round(3)
    return out


# --- build (I/O) ------------------------------------------------------------ #
def split_paths(out_dir: str = "data/vi_mix") -> dict:
    return {s: os.path.join(out_dir, s) for s in _SPLITS}


def heldout_path(name: str, out_dir: str = "data/vi_mix") -> str:
    return os.path.join(out_dir, f"heldout_{name}")


def _version_path(out_dir: str) -> str:
    return os.path.join(out_dir, "MIX_VERSION")


def cached_version(out_dir: str = "data/vi_mix") -> int:
    """Version stamp of the cache on disk; 0 if absent or unreadable."""
    try:
        with open(_version_path(out_dir)) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return 0


def prepare_mixture(out_dir: str = "data/vi_mix", token: str | None = None) -> dict:
    """Build (or reuse) the cached multi-source training mixture.

    Writes ``out_dir/{train,val,test}`` in ``data_prep``'s record schema, plus
    ``out_dir/heldout_<source>`` for every hash-split source so its benchmark
    can be scored on rows the mixture never trained on.
    """
    import soundfile as sf
    from datasets import Dataset
    from tqdm.auto import tqdm

    import bench
    import data_prep

    paths = split_paths(out_dir)
    heldouts = {n: heldout_path(n, out_dir) for n, s in MIX.items() if s.hash_split}
    wanted = list(paths.values()) + list(heldouts.values())
    on_disk = all(os.path.exists(os.path.join(p, "dataset_info.json")) for p in wanted)
    have = cached_version(out_dir)
    if on_disk and have == MIX_VERSION:
        print(f"[mixture] cache hit at {out_dir} (v{MIX_VERSION}); skipping build.")
        return paths
    if on_disk:
        # Without this the notebook would print "cache hit" and quietly retrain
        # on the very transcripts a version bump exists to fix.
        print(f"[mixture] cache at {out_dir} is v{have}, need v{MIX_VERSION} — "
              "rebuilding (~1 h, re-streams the non-viVoice sources).")

    token = token or os.environ.get("HF_TOKEN")
    records: dict[str, list] = {s: [] for s in _SPLITS}

    # 1. viVoice: reuse the existing channel-disjoint cache in place. The wav
    #    files already exist, so we only copy the lightweight metadata rows.
    vivoice_dir = MIX["vivoice"].cached_dir
    cached = data_prep.load_splits(vivoice_dir)
    for split in _SPLITS:
        for row in cached[split]:
            records[split].append({**row, "text": normalize_train_text(row["text"]),
                                   "source": "vivoice"})
    print(f"[mixture] vivoice: reused {sum(len(cached[s]) for s in _SPLITS)} rows "
          f"from {vivoice_dir}")

    # 2. Streamed sources: decode, resample and write 16 kHz wavs.
    audio_root = os.path.join(out_dir, "audio")
    for name, src in MIX.items():
        if src.repo is None:
            continue
        dest = os.path.join(audio_root, name)
        os.makedirs(dest, exist_ok=True)
        written = [0]

        def emit(rows, forced_split, desc):
            for row in tqdm(rows, desc=desc, unit="clip"):
                text = normalize_train_text(row["text"])
                # hash_bucket lowercases its key, so case normalization leaves
                # every row in the split it was already assigned to — the VLSP
                # held-out slice keeps its identity across this version bump.
                split = hash_bucket(text) if src.hash_split else forced_split
                wav = os.path.abspath(
                    os.path.join(dest, f"{name}_{written[0]:06d}.wav"))
                sf.write(wav, row["array"], TARGET_SR)
                written[0] += 1
                records[split].append({
                    "audio_path": wav,
                    "text": text,
                    "duration": row["duration"],
                    # These sources are split officially or by transcript hash,
                    # so channel is provenance only, not a splitting key.
                    "channel": name,
                    "bucket": data_prep._bucket(row["duration"]),
                    "source": name,
                })

        emit(take_until_hours(
                bench.stream_rows(src.repo, src.train_split, src.config,
                                  src.text_column, token),
                src.hours),
             "train", f"{name}:{src.train_split}")
        if src.val_split:
            emit(take_until_hours(
                    bench.stream_rows(src.repo, src.val_split, src.config,
                                      src.text_column, token),
                    VAL_HOURS_CAP),
                 "val", f"{name}:{src.val_split}")

    # 3. Save the mixture, then each hash-split source's held-out slice.
    os.makedirs(out_dir, exist_ok=True)
    for split in _SPLITS:
        Dataset.from_list(records[split]).save_to_disk(paths[split])
        print(f"[mixture] {split}: {len(records[split])} rows -> {paths[split]}")
    for name, path in heldouts.items():
        rows = [r for r in records["test"] if r["source"] == name]
        Dataset.from_list(rows).save_to_disk(path)
        print(f"[mixture] heldout {name}: {len(rows)} rows -> {path}")

    # Written last: a build that dies half-way leaves no stamp, so the next run
    # rebuilds rather than trusting a partial cache.
    with open(_version_path(out_dir), "w") as fh:
        fh.write(str(MIX_VERSION))

    print("\n[mixture] training composition:")
    print(summarize(records["train"]).to_string(index=False))
    return paths
