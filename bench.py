"""External Vietnamese ASR benchmarks (VIVOS, Common Voice, VLSP 2020).

Companion to ``data_prep.py``, which handles the viVoice corpus this project
fine-tunes on. This module streams *published* benchmark sets so the baseline
can be placed next to the numbers reported for other Vietnamese ASR models.

On the PhoWhisper comparison
----------------------------
PhoWhisper's reported WER uses its own text normalization, which is not fully
specified, so ``phowhisper_large`` below is a **reference point, not a
like-for-like baseline**. Treat a gap of a point or two as noise.

On VLSP 2020
------------
We deliberately do not chase PhoWhisper's Task-1/Task-2 columns: those test sets
are not public. Instead ``doof-ferb/vlsp2020_vinai_100h`` (the VinBigData 100h
release) is used as a corpus in its own right — the mixture trains on 90% of it
and this module scores the 5% held-out slice, split by transcript hash so the two
never overlap. That is a self-consistent benchmark, just not the same number
PhoWhisper reports, so it carries no reference score.

One caveat survives: the publisher estimates transcript accuracy at ~96%, so a
few points of the measured WER are label noise rather than model error. Read the
before/after *delta* on this set, not its absolute value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Column names seen across the mirrors, most-specific first.
_TEXT_COLUMNS = ("transcription", "sentence", "transcript", "text",
                 "original_sentence")
_AUDIO_COLUMNS = ("audio",)

TARGET_SR = 16_000


@dataclass(frozen=True)
class Spec:
    """One benchmark: where to stream it from and how to read it."""

    repo: str | None = None             # HF dataset id; None -> local_dir
    split: str = "test"
    local_dir: str | None = None        # locally built held-out slice
    config: str | None = None
    text_column: str | None = None      # None -> autodetect
    limit: int | None = None            # None -> the whole split
    shuffle: bool = False               # seeded subsample before taking `limit`
    phowhisper_large: float | None = None   # published WER %, reference only
    note: str = ""


BENCHMARKS: dict[str, Spec] = {
    # Official VIVOS test split (760 utterances). AILAB-VNUHCM/vivos is a
    # loading-script repo, which datasets>=4 no longer runs, so we use a parquet
    # mirror and read the untouched official transcript.
    "vivos": Spec(
        repo="htdung167/vivos-preprocessed-v2",
        split="test",
        text_column="original_sentence",
        phowhisper_large=4.67,
        note="Official VIVOS test split, 760 utterances.",
    ),
    # Common Voice 17.0 Vietnamese test split (1274 utterances). The
    # mozilla-foundation repo is script-based and gated; fixie-ai mirrors it as
    # parquet with identical splits.
    "cmv_vi": Spec(
        repo="fixie-ai/common_voice_17_0",
        split="test",
        config="vi",
        text_column="sentence",
        phowhisper_large=8.14,
        note="Common Voice 17.0 Vietnamese test split, 1274 utterances.",
    ),
    # See the module docstring: training corpus, no reference score. The
    # mixture now trains on this corpus, so we evaluate on the hash-assigned
    # held-out slice that mixture.prepare_mixture() writes -- never on a sample
    # of the train stream, which would be training on the test set.
    "vlsp2020_100h": Spec(
        local_dir="data/vi_mix/heldout_vlsp2020_100h",
        split="test",
        phowhisper_large=None,
        note=("VLSP 2020 VinBigData 100h corpus, 5% held-out slice assigned by "
              "transcript hash. Own benchmark, not PhoWhisper's T1/T2. "
              "~96% transcript accuracy -> read the delta, not the absolute."),
    ),
}


# --- column detection ------------------------------------------------------- #
def pick_text_column(columns, override: str | None = None) -> str:
    columns = list(columns)
    if override is not None:
        if override not in columns:
            raise KeyError(f"text column {override!r} not in {columns}")
        return override
    for candidate in _TEXT_COLUMNS:
        if candidate in columns:
            return candidate
    raise KeyError(f"no known text column in {columns}")


def pick_audio_column(columns) -> str:
    columns = list(columns)
    for candidate in _AUDIO_COLUMNS:
        if candidate in columns:
            return candidate
    raise KeyError(f"no known audio column in {columns}")


def duration_s(array, sr: int) -> float:
    return len(array) / float(sr)


# --- streaming loader (network) --------------------------------------------- #
def stream_rows(repo: str, split: str, config: str | None = None,
                text_column: str | None = None, token: str | None = None,
                limit: int | None = None, shuffle_seed: int | None = None,
                extra_columns: tuple = ()):
    """Yield ``{"array", "text", "duration", "path"}`` from an HF audio dataset, 16 kHz.

    Streams, so a subsampled source never downloads the full corpus. Shared by
    the benchmark loader and the training mixture.

    ``path`` is the in-archive filename (``""`` when the source does not carry
    one). Most callers ignore it; ``corpora.py`` needs it because VietSpeech
    encodes its recording id in the filename and has no speaker column.

    ``extra_columns`` names source columns to pass through untouched — how
    ``corpora.py`` gets VieNeu's ``speaker`` field. Missing names yield ``None``
    rather than raising, so a source that drops a column degrades to "no
    grouping signal" instead of killing a multi-hour build.
    """
    from datasets import Audio, load_dataset

    import data_prep

    ds = load_dataset(repo, config, split=split, streaming=True, token=token)
    audio_col = pick_audio_column(ds.features)
    text_col = pick_text_column(ds.features, text_column)
    # Decode the bytes ourselves: the datasets audio decoder goes through
    # torchcodec+FFmpeg, and FFmpeg 4 on this machine is buggy (see README).
    ds = ds.cast_column(audio_col, Audio(decode=False))

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=max(1000, limit or 1000))
    if limit is not None:
        ds = ds.take(limit)

    yield from _decode_rows(ds, audio_col, text_col, extra_columns)


def _decode_rows(ds, audio_col: str, text_col: str, extra_columns: tuple = ()):
    """Turn raw dataset rows into the ``{"array", "text", "duration", "path"}`` shape.

    Split out of ``stream_rows`` so ``stream_shards`` produces byte-identical
    rows — the two differ only in how the bytes arrive, and a build must not be
    able to tell which path fetched it.
    """
    import data_prep

    for row in ds:
        entry = row[audio_col]
        arr, sr = data_prep._decode_audio(entry)
        if sr != TARGET_SR:
            arr = data_prep._resample(arr, sr, TARGET_SR)
            sr = TARGET_SR
        out = {"array": arr, "text": row[text_col],
               "duration": duration_s(arr, sr),
               "path": (entry.get("path") or "") if isinstance(entry, dict) else ""}
        for col in extra_columns:
            out[col] = row.get(col)
        yield out


# --- shard prefetch --------------------------------------------------------- #
# Why this exists, and why it is the default for corpus builds:
#
# `stream_rows` holds one long HTTP read open through `HfFileSystem`. On a lossy
# link that read dies, and the retry cannot recover it -- the reset closes the
# underlying httpx client, so `http_backoff`'s own "Retry 1/5" immediately
# raises `RuntimeError: Cannot send a request, as the client has been closed`.
# Measured on this box: a reset every 35-70 min, every one of them fatal, against
# a 100 h corpus that needs ~78 min of continuous streaming. Three consecutive
# attempts at viVoice lost 92%, 55% and counting.
#
# Fetching whole shard files instead fixes the failure mode rather than papering
# over it: each file is an independent, resumable download, a reset costs one
# file, and the retry below builds a *fresh* client so the closed-client bug
# cannot recur. Concurrency is the free side effect -- several connections where
# there was one, which matters because a single stream measured 19 Mbit/s.
_DATA_SUFFIXES = (".parquet", ".arrow")
_FETCH_ATTEMPTS = 4
# The window is what we may over-fetch: the caller stops mid-corpus once its
# hour budget is met, and anything already in flight is wasted. Bounded by bytes
# as well as count because shard sizes differ by an order of magnitude across
# these corpora (viVoice 476 MB, VietSpeech 4.8 GB, VieNeu 485 MB).
#
# Keep this at 2. The bound that matters is RAM, not bytes on the wire:
# `hf_xet` reconstructs each file from chunks in memory, so N concurrent shards
# means N reconstruction buffers. At 8 wide this box (15 GB) was OOM-killed
# twice -- 7.6 GB RSS with `hf-xet-12` invoking the killer, then 9.2 GB -- once
# on viVoice's 476 MB shards and once on VieNeu's 485 MB.
#
# Two wide costs nothing to pay for that. Measured: VietSpeech built 100 h in 20
# min at 2 wide, *faster* than viVoice's 27 min at 8. The link saturates well
# before the window does, because `hf_xet` already parallelises within each file
# -- 2 files is already many connections. Against the single-stream reader this
# replaced (2.4 MB/s, 82 audio-min per wall-min) it is still ~5x.
#
# The byte bound stays as a backstop for large shards: VietSpeech's are 4.8 GB,
# so even 2 is 9.6 GB in flight.
_PREFETCH_WORKERS = 2
_PREFETCH_BYTES = 4e9


def split_shards(files, split: str) -> list[str]:
    """The data files belonging to ``split``, in stable order.

    Two layouts in play. viVoice and VietSpeech name the split in the file
    (``data/train-00000-of-00354.parquet``); VieNeu ships a `save_to_disk` tree
    whose files carry no split marker at all (``data-00000-of-00049.arrow``) and
    *is* one split. So an unmatched filter means "single-split repo", not "no
    data" -- returning [] there would silently build an empty corpus.
    """
    import re

    data = sorted(f for f in files if f.endswith(_DATA_SUFFIXES))
    marked = [f for f in data if re.search(rf"(^|/){re.escape(split)}[-_/]", f)]
    return marked or data


def _fetch_shard(repo: str, filename: str, token: str | None) -> str:
    """Download one shard to the HF cache, returning its local path.

    The retry is the whole point of the module-level comment: `hf_hub_download`
    is called afresh each attempt, so a connection reset cannot leave us holding
    a dead client. It also resumes from the partial file, so a late failure on a
    4 GB shard does not restart it from zero.
    """
    import time

    from huggingface_hub import hf_hub_download

    for attempt in range(1, _FETCH_ATTEMPTS + 1):
        try:
            return hf_hub_download(repo_id=repo, filename=filename,
                                   repo_type="dataset", token=token)
        except Exception as exc:                      # noqa: BLE001
            if attempt == _FETCH_ATTEMPTS:
                raise
            wait = min(30, 2 ** attempt)
            print(f"[bench] {filename}: {type(exc).__name__} — retry "
                  f"{attempt}/{_FETCH_ATTEMPTS - 1} in {wait}s", flush=True)
            time.sleep(wait)


def _prefetch_window(repo: str, shards: list[str], token: str | None,
                     workers: int) -> int:
    """How many shards to keep in flight: ``workers``, capped by bytes."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).get_paths_info(repo, shards[:1],
                                                 repo_type="dataset")
        size = getattr(info[0], "size", None) if info else None
    except Exception:                                 # noqa: BLE001 - sizing is
        size = None                                   # an optimisation, not a
    if not size:                                      # requirement
        return max(1, min(workers, 2))
    return max(1, min(workers, int(_PREFETCH_BYTES // size) or 1))


def stream_shards(repo: str, split: str, config: str | None = None,
                  text_column: str | None = None, token: str | None = None,
                  extra_columns: tuple = (), shuffle_seed: int | None = None,
                  workers: int = _PREFETCH_WORKERS):
    """Yield the same rows as ``stream_rows``, but shard-by-shard off disk.

    Downloads run ``workers``-wide in a sliding window ahead of the consumer, so
    the network stays busy while rows are being decoded and written. The caller
    is expected to stop early (an hour budget); shards past that point are never
    requested, and the ones in flight are the only waste.

    ``config`` is accepted for signature parity with ``stream_rows`` and must be
    None -- resolving a named config to its shard subset is layout-specific, and
    none of the corpora in ``corpora.CORPORA`` uses one. Passing one raises
    rather than quietly building from the wrong files.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi

    if config is not None:
        raise ValueError(
            f"stream_shards cannot resolve config={config!r} to shard files; "
            "use stream_rows for configured datasets")

    shards = split_shards(
        HfApi(token=token).list_repo_files(repo, repo_type="dataset"), split)
    if not shards:
        raise ValueError(f"{repo}: no {_DATA_SUFFIXES} data files found")
    fmt = "parquet" if shards[0].endswith(".parquet") else "arrow"

    if shuffle_seed is not None:
        # Shard order is the part of streaming `shuffle` that matters here: it is
        # what stops an hour budget from always being drawn from the same prefix
        # of the corpus (VieNeu is ordered by voice, so a prefix is a handful of
        # speakers). The in-shard buffer shuffle is applied per shard below.
        import random

        random.Random(shuffle_seed).shuffle(shards)

    window = _prefetch_window(repo, shards, token, workers)
    print(f"[bench] {repo}: {len(shards)} {fmt} shard(s) for split {split!r}, "
          f"prefetching {window} at a time", flush=True)

    pool = ThreadPoolExecutor(max_workers=workers)
    pending: deque = deque()
    nxt = 0
    try:
        while nxt < min(window, len(shards)):
            pending.append((shards[nxt], pool.submit(_fetch_shard, repo,
                                                     shards[nxt], token)))
            nxt += 1
        while pending:
            name, fut = pending.popleft()
            path = fut.result()
            if nxt < len(shards):                     # refill as one is consumed
                pending.append((shards[nxt], pool.submit(_fetch_shard, repo,
                                                         shards[nxt], token)))
                nxt += 1
            ds = load_dataset(fmt, data_files=[path], split="train",
                              streaming=True)
            audio_col = pick_audio_column(ds.features)
            text_col = pick_text_column(ds.features, text_column)
            # Same reason as stream_rows: decode the bytes ourselves rather than
            # through torchcodec+FFmpeg.
            ds = ds.cast_column(audio_col, Audio(decode=False))
            if shuffle_seed is not None:
                ds = ds.shuffle(seed=shuffle_seed, buffer_size=1000)
            yield from _decode_rows(ds, audio_col, text_col, extra_columns)
    finally:
        # The consumer stops early by design, which closes this generator with
        # downloads still running. Without cancel_futures they would keep pulling
        # gigabytes nobody will read.
        pool.shutdown(wait=False, cancel_futures=True)


def iter_bench(name: str, token: str | None = None, limit: int | None = None,
               seed: int = 42):
    """Yield ``{"array", "text", "duration"}`` dicts for one benchmark.

    ``limit`` overrides the registry value (handy for smoke tests).
    """
    spec = BENCHMARKS[name]
    if spec.local_dir is not None:      # held-out slice built by mixture.py
        yield from _iter_local(spec, limit)
        return
    yield from stream_rows(
        repo=spec.repo, split=spec.split, config=spec.config,
        text_column=spec.text_column, token=token,
        limit=spec.limit if limit is None else limit,
        shuffle_seed=seed if spec.shuffle else None,
    )


def _iter_local(spec: "Spec", limit: int | None = None):
    """Yield rows from a locally cached split (schema written by data_prep)."""
    import data_prep
    from datasets import load_from_disk

    if not os.path.exists(os.path.join(spec.local_dir, "dataset_info.json")):
        raise FileNotFoundError(
            f"held-out slice missing at {spec.local_dir}. Build it first:\n"
            "    import mixture; mixture.prepare_mixture()\n"
            "(it is written alongside the training mixture, so notebook 2's "
            "data cell must have run at least once)")
    ds = load_from_disk(spec.local_dir)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        arr, sr = data_prep.read_audio(row["audio_path"])
        yield {"array": arr, "text": row["text"], "duration": duration_s(arr, sr)}


# --- running a benchmark ---------------------------------------------------- #
def length_sorted_batches(lengths, batch_size: int) -> list[list[int]]:
    """Index batches ordered by length, so each batch pads efficiently.

    Returns *original* indices; callers must scatter results back by index
    rather than appending, or hypotheses silently misalign with references.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[s:s + batch_size] for s in range(0, len(order), batch_size)]


def transcribe_arrays(model, processor, arrays, batch_size: int = 8,
                      max_new_tokens: int = 440, language: str = "Vietnamese",
                      desc: str = "transcribing (batched)"):
    """Batched greedy transcription of in-memory 16 kHz arrays.

    Shared by both notebooks. ``processor.tokenizer.padding_side`` must be
    ``"left"`` for batched generation to be correct.
    """
    import torch
    from tqdm.auto import tqdm

    hyps = [None] * len(arrays)
    batches = length_sorted_batches([len(a) for a in arrays], batch_size)
    with torch.no_grad():
        for chunk in tqdm(batches, desc=desc):
            inputs = processor.apply_transcription_request(
                audio=[arrays[i] for i in chunk], language=language,
            ).to(model.device, model.dtype)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
            gen = out[:, inputs["input_ids"].shape[1]:]
            decoded = processor.decode(gen, return_format="transcription_only")
            for i, txt in zip(chunk, decoded):
                hyps[i] = txt
    return hyps


def run_benchmarks(model, processor, names=None, token: str | None = None,
                   batch_size: int = 8, limit: int | None = None):
    """Evaluate ``model`` on each benchmark; return (metrics, prediction frames)."""
    import pandas as pd
    from tqdm.auto import tqdm

    from vi_norm import wer_cer

    names = list(BENCHMARKS) if names is None else list(names)
    results, frames = {}, {}
    for name in names:
        spec = BENCHMARKS[name]
        print(f"\n=== {name} · {spec.repo or spec.local_dir} [{spec.split}] ===")
        print("   ", spec.note)
        rows = list(tqdm(iter_bench(name, token=token, limit=limit),
                         desc=f"loading {name}"))
        hyps = transcribe_arrays(model, processor, [r["array"] for r in rows],
                                 batch_size=batch_size)
        frames[name] = pd.DataFrame({
            "dataset": name,
            "duration": [round(r["duration"], 2) for r in rows],
            "ref": [r["text"] for r in rows],
            "hyp": hyps,
        })
        # Record where the rows came from: it makes a stale baseline file
        # (e.g. VLSP scored before the held-out slice existed) detectable.
        results[name] = {**wer_cer(frames[name]["ref"], frames[name]["hyp"]),
                         "source": spec.repo or spec.local_dir}
        m = results[name]
        print(f"   WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
              f"(legacy WER={m['wer_legacy']:.4f})  n={m['n']}")
    return results, frames


# --- reporting -------------------------------------------------------------- #
def compare_table(results: dict[str, dict]):
    """Build the summary table, in registry order.

    ``results`` maps benchmark name -> the dict returned by ``vi_norm.wer_cer``.
    """
    import pandas as pd

    unknown = set(results) - set(BENCHMARKS)
    if unknown:
        raise KeyError(f"unknown benchmark(s): {sorted(unknown)}")

    rows = []
    for name, spec in BENCHMARKS.items():
        if name not in results:
            continue
        m = results[name]
        rows.append({
            "dataset": name,
            "n": m["n"],
            "WER %": round(100 * m["wer"], 2),
            "CER %": round(100 * m["cer"], 2),
            "WER % (legacy)": round(100 * m["wer_legacy"], 2),
            "PhoWhisper-large": (spec.phowhisper_large
                                 if spec.phowhisper_large is not None else ""),
        })
    return pd.DataFrame(rows)
