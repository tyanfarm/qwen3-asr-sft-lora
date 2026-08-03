"""Three large Vietnamese speech corpora, cached on one schema.

``data_prep.py`` handles viVoice specifically and ``mixture.py`` assembles the
small multi-source mixture used by notebooks 1-2. This module is for the
*large-corpus* track (notebooks 3-4): viVoice, VieNeu-TTS-1000h and VietSpeech,
each streamed to a capped number of hours, split into train/val/test, and
written in exactly the record schema ``data_prep`` uses -- ``audio_path``,
``text``, ``duration``, ``channel``, ``bucket``, plus ``source``. That means
``data_prep.load_splits``, ``data_prep.read_audio`` and the notebooks' collator
all work against these caches unchanged.

Why a separate module from ``mixture.py``
-----------------------------------------
``mixture.py`` builds one blended training set and holds out slices of *public
benchmarks*. Here each corpus keeps its own train/val/test so the fine-tune can
be scored per corpus -- the question these notebooks ask is "does more data from
these three sources move each corpus's own held-out WER", which needs three
separate test sets, not one blend.

Keeping the test splits honest
------------------------------
None of the three ships an official test split, so every split is built here.
The rule is *never split at the clip level*: consecutive clips from one
recording share a speaker, a room and a topic, so a random clip-level split
leaks the test set into training and reports a WER that is far too good.

* **viVoice** carries a real ``channel`` column and is already split
  channel-disjoint by ``data_prep.split_by_channel``. This module reuses that
  cache rather than rebuilding it, so notebook 3's viVoice number stays
  comparable with ``results/baseline_metrics.json``.
* **VietSpeech** has no speaker column -- only ``audio`` and ``transcription``.
  Its filenames are ``<recording>_<clip>.wav`` (``278_000000086.wav``), and a
  3,000-row sample showed 487 distinct prefixes with up to 40 clips each, so the
  prefix groups clips by source recording. Splitting on it is weaker than true
  speaker-disjointness (one speaker may appear under several recordings) but it
  removes the dominant leak, which is same-recording neighbours.
* **VieNeu** carries a real ``speaker`` column, but it holds one id *per
  recording* (``jellyfish1010_0041``), not per voice -- the card counts 193
  voices across 74,858 clips. ``channel_strip_suffix`` drops the trailing index
  so all of one voice's recordings share a channel and cannot straddle splits.

On the two VieNeu releases
--------------------------
This uses ``pnnbao-ump/VieNeu-TTS-140h``, not the 1000-hour sibling. The 1000 h
release is gated ``manual`` and its authors restrict it to institutions; this
project's request is still awaiting review, so every file resolves 403. The
140 h release is gated ``auto`` -- accepting the terms on the dataset page is
enough -- and is otherwise the same corpus shape.

It ships **both** ``text`` (orthographic Vietnamese) and ``phonemized_text``
(IPA). ``text_column`` is pinned to ``text`` deliberately: phonemes are the
wrong target here, because the model is scored on words. Audio is 24 kHz and is
resampled to 16 kHz by ``bench.stream_rows`` on the way in.

VieNeu's voice count is shard-bound, not hour-bound
---------------------------------------------------
Its 193 voices sit contiguously across 49 shards -- about 4 voices per 2.9 h
shard. Measured: a 0.4 h build saw 4 voices and a **3 h build saw 5**. Raising
``target_hours`` buys clips far faster than it buys speakers, and because the
split is speaker-disjoint it is *voice* count that decides whether the test WER
means anything. Budget roughly 3 h per extra voice, and read a VieNeu test WER
built on a couple of voices as describing those speakers, not the corpus.
``allocate_channels`` keeps such a split buildable and ``prepare_corpus`` warns
when a corpus lands under 10 channels; notebook 3 prints per-split channel counts.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

TARGET_SR = 16_000

# Bumped whenever a change alters the *content* of a cached corpus, so a stale
# data/corpora/<name>/ is rebuilt rather than silently reused.
#   v2 - wavs are written into audio/<bucket>/ subdirectories (see AUDIO_SHARD).
#   v3 - the stamp also records target_hours, so raising it rebuilds instead of
#        hitting the smaller cache (see cached_stamp).
CORPUS_VERSION = 3

# Wavs per subdirectory. A whole-corpus build of VietSpeech is ~940k clips, and
# a single directory with that many entries makes every later ls/stat crawl.
AUDIO_SHARD = 1000

SPLITS = ("train", "val", "test")

# Channel-disjoint split of the channel list, matching data_prep's default.
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# Below this a corpus cannot be split three ways without a channel spanning two
# splits, which would leak the test set into training.
MIN_CHANNELS = 3

# Clips longer than this are dropped at build time. The measured peak on the
# 16 GB box was 6.7 GB at ~57 s; past that is untested territory, and a single
# over-long clip taking out a multi-hour run is not a trade worth making.
MAX_CLIP_S = 60.0

# Sub-5s clips are kept here (unlike data_prep's merged viVoice segments, which
# drop them): VietSpeech is mostly 2-6 s and dropping them would discard most of
# the corpus. See the notebooks for what that does to the duration mixture.
MIN_CLIP_S = 0.5


@dataclass(frozen=True)
class Corpus:
    """One large corpus: where to stream it and how to group it into channels."""

    repo: str | None = None            # HF dataset id; None -> already cached
    cached_dir: str | None = None      # prebuilt cache to reuse as-is
    config: str | None = None
    split: str = "train"
    text_column: str | None = None     # None -> autodetect via bench.pick_text_column
    channel_column: str | None = None  # column holding a speaker/recording id
    channel_from_path: bool = False    # derive the channel from the audio filename
    channel_strip_suffix: bool = False  # drop a trailing _<index> from the channel id
    shuffle: bool = False              # shuffle the stream before taking the hour budget
    gated: bool = False                # needs an approved access request
    note: str = ""


CORPORA: dict[str, Corpus] = {
    # Reused in place: notebook 1 built this with channel-disjoint splits, and
    # rebuilding it would change the channel shuffle and invalidate the existing
    # baseline. To train on more viVoice, build a second cache with
    # data_prep.prepare_dataset(target_hours=N, out_dir=...) and point here.
    "vivoice": Corpus(
        cached_dir="data/vi_asr",
        note=("capleaf/viVoice, reused from data_prep's cache. Consecutive "
              "same-channel clips merged into 5-60 s segments; split by channel."),
    ),
    # The whole viVoice corpus as its native clips. Separate from "vivoice"
    # because the two cannot be the same entry: data_prep's merged build holds
    # every decoded clip in RAM before writing (~230 GB at 1,000 h), so it
    # cannot scale past the 8 h cache, while this path writes incrementally and
    # holds only metadata. The cost is that clips arrive unmerged -- no 5-60 s
    # long-form segments -- which is what the source actually ships.
    "vivoice_full": Corpus(
        repo="capleaf/viVoice",
        split="train",
        text_column="text",
        channel_column="channel",
        gated=True,
        note=("capleaf/viVoice, ~1,000 h of YouTube speech as native clips "
              "(unmerged). Split by its real `channel` column."),
    ),
    "vietspeech": Corpus(
        repo="NhutP/VietSpeech",
        split="train",
        text_column="transcription",
        channel_from_path=True,
        note=("NhutP/VietSpeech, ~1,100 h of Vietnamese social media speech "
              "(north/central/south accents). Native 16 kHz, lowercase, "
              "unpunctuated. Split by filename recording prefix."),
    ),
    "vieneu": Corpus(
        repo="pnnbao-ump/VieNeu-TTS-140h",
        split="train",
        # Must be explicit: the corpus also ships `phonemized_text` (IPA), which
        # is the wrong target -- this model is scored on words, not phonemes.
        text_column="text",
        channel_column="speaker",
        channel_strip_suffix=True,
        # The stream is ordered by voice, so a contiguous prefix is a handful of
        # speakers: a 0.4 h probe gave 222 clips across 4 voices and an empty val
        # split. Shuffling randomises shard order, so the hour budget is drawn
        # from across the corpus rather than always its first shards. It does
        # NOT manufacture voice diversity -- the same probe shuffled still saw
        # only 5 voices, because 0.4 h fits inside one shard. Voice count scales
        # with hours here; see the note about target_hours below.
        shuffle=True,
        gated=True,
        note=("pnnbao-ump/VieNeu-TTS-140h, 74,858 studio TTS clips (~140.7 h, "
              "193 voices, 24 kHz -> resampled to 16 kHz). Gated 'auto', so "
              "accepting the terms on the dataset page is enough. Split by "
              "voice (the `speaker` column with its per-recording suffix "
              "stripped)."),
    ),
}


# --- pure helpers (unit-tested, no I/O) ------------------------------------- #
def strip_index_suffix(value: str) -> str:
    """Drop a trailing ``_<index>`` so per-recording ids collapse to one voice.

    ``jellyfish1010_0041`` -> ``jellyfish1010``. Left unchanged when there is no
    underscore. VieNeu's ``speaker`` column is one entry *per recording*, not
    per voice, so grouping on it directly would let the same voice appear in
    both train and test -- the exact leak the splits exist to prevent.
    """
    return value.rsplit("_", 1)[0] if "_" in value else value


def path_channel(path: str | None) -> str:
    """Group key from an audio filename, for corpora with no speaker column.

    VietSpeech names clips ``<recording>_<clip>.wav``, so everything before the
    last underscore identifies the source recording. Falls back to the whole
    stem when there is no underscore, and to ``"unknown"`` when there is no
    path at all -- which collapses every such row into one channel, so they all
    land in the same split rather than leaking across splits one by one.
    """
    if not path:
        return "unknown"
    stem = os.path.splitext(os.path.basename(path))[0]
    if not stem:
        return "unknown"
    return strip_index_suffix(stem)


def channel_of(row: dict, corpus: Corpus) -> str:
    """The split-grouping key for one streamed row."""
    if corpus.channel_column:
        raw = row.get(corpus.channel_column)
        if raw is None or str(raw) == "":
            # Degrade to one shared channel rather than a unique key per row:
            # unique keys would scatter these rows across all three splits.
            return "unknown"
        raw = str(raw)
        return strip_index_suffix(raw) if corpus.channel_strip_suffix else raw
    if corpus.channel_from_path:
        return path_channel(row.get("path"))
    # No grouping signal at all: fall back to the transcript hash, which at
    # least keeps duplicate sentences out of two splits (see mixture.py).
    return ""


def has_channel_signal(corpus: Corpus) -> bool:
    """Whether this corpus can be split channel-disjoint rather than by hash."""
    return bool(corpus.channel_column or corpus.channel_from_path)


def keep_clip(duration: float, text: str) -> bool:
    """Drop empty transcripts and clips outside the trainable duration range."""
    return bool(text and text.strip()) and MIN_CLIP_S <= duration <= MAX_CLIP_S


def allocate_channels(channels, rng, ratios=SPLIT_RATIOS) -> dict[str, set]:
    """Split a channel list three ways, never starving val or test.

    ``data_prep.split_by_channel`` rounds to the ratios and drops val to zero
    when few channels remain -- fine for viVoice's 175 channels, fatal here.
    VieNeu stores its 193 voices contiguously across 49 shards, so an hour
    budget buys roughly 4-5 voices per 2.9 h shard however large it is: a 3 h
    build saw 5 voices and a 0.4 h build saw 4. Under plain rounding both give
    an empty val split, and an empty split cannot be loaded back at all.

    Val and test therefore get at least one channel each, and train takes the
    rest. Deliberately *not* a change to ``data_prep.split_by_channel``: that
    function's exact output defines the viVoice splits notebooks 1-2 already
    published results against.
    """
    channels = sorted(channels)
    rng.shuffle(channels)
    n = len(channels)
    if n < MIN_CHANNELS:
        raise ValueError(
            f"only {n} channel(s); need at least {MIN_CHANNELS} to build "
            "train/val/test without a channel spanning two splits")
    n_val = max(1, round(ratios[1] * n))
    n_test = max(1, round(ratios[2] * n))
    n_train = n - n_val - n_test
    assert n_train >= 1, (n, n_val, n_test)
    return {
        "val": set(channels[:n_val]),
        "test": set(channels[n_val:n_val + n_test]),
        "train": set(channels[n_val + n_test:]),
    }


def split_records(records: list[dict], corpus: Corpus, rng) -> dict[str, list]:
    """Assign records to train/val/test without splitting a recording.

    Channel-disjoint when the corpus carries a grouping signal, otherwise by
    transcript hash. Both guarantee the same thing at different strengths: a
    unit of speech never appears in two splits.
    """
    import mixture

    out: dict[str, list] = {s: [] for s in SPLITS}
    if has_channel_signal(corpus):
        owner = allocate_channels({r["channel"] for r in records}, rng)
        where = {ch: name for name, chans in owner.items() for ch in chans}
        for row in records:
            out[where[row["channel"]]].append(row)
        return out
    for row in records:
        out[mixture.hash_bucket(row["text"])].append(row)
    return out


def corpus_dir(name: str, root: str = "data/corpora") -> str:
    """Where a built corpus lives. viVoice reports its prebuilt cache instead."""
    cached = CORPORA[name].cached_dir
    return cached if cached else os.path.join(root, name)


def split_paths(name: str, root: str = "data/corpora") -> dict:
    base = corpus_dir(name, root)
    return {s: os.path.join(base, s) for s in SPLITS}


def _version_path(base: str) -> str:
    return os.path.join(base, "CORPUS_VERSION")


def hours_tag(target_hours: float | None) -> str:
    """How ``target_hours`` is recorded in the stamp. ``None`` -> ``"all"``."""
    return "all" if target_hours is None else f"{target_hours:g}"


def cached_stamp(base: str) -> tuple[int, str | None]:
    """``(version, hours_tag)`` of the cache on disk; ``(0, None)`` if unreadable.

    The stamp carries the hours the cache was built with because the version
    alone cannot tell a 100 h build from a whole-corpus one -- they are both
    v2. Without it, raising ``target_hours`` would silently register as a cache
    hit and train on the old, smaller corpus. A stamp with no hours field is a
    pre-v3 cache; it reports ``None``, which matches no request and so rebuilds.
    """
    try:
        with open(_version_path(base)) as fh:
            parts = fh.read().split()
        return int(parts[0]), (parts[1] if len(parts) > 1 else None)
    except (OSError, ValueError, IndexError):
        return 0, None


# --- access probing --------------------------------------------------------- #
def check_access(name: str, token: str | None = None) -> tuple[bool, str]:
    """Try to pull a single row; return ``(ok, message)``.

    A gated repo answers 200 to the metadata API whether or not the request has
    been approved, so listing files proves nothing -- only fetching content
    does. Pulling one row is the cheapest honest check, and it is what lets the
    notebooks skip an unapproved corpus with an explanation rather than dying
    an hour into a build.
    """
    corpus = CORPORA[name]
    if corpus.repo is None:
        base = corpus_dir(name)
        ok = all(os.path.exists(os.path.join(base, s, "dataset_info.json"))
                 for s in SPLITS)
        return ok, (f"cached at {base}" if ok else
                    f"no cache at {base} — run data_prep.prepare_dataset() first")

    token = token or os.environ.get("HF_TOKEN")
    try:
        from datasets import load_dataset

        ds = load_dataset(corpus.repo, corpus.config, split=corpus.split,
                          streaming=True, token=token)
        row = next(iter(ds))
        return True, f"{corpus.repo}: ok, columns {sorted(row)}"
    except Exception as exc:                       # noqa: BLE001 - report anything
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return False, f"{corpus.repo}: {type(exc).__name__}: {first[:200]}"


# --- build (I/O) ------------------------------------------------------------ #
def prepare_corpus(name: str, target_hours: float | None = 10.0,
                   root: str = "data/corpora", token: str | None = None,
                   seed: int = 42) -> dict:
    """Stream ``target_hours`` of one corpus and cache train/val/test splits.

    ``target_hours=None`` takes the whole corpus. Budget for it: cached audio
    costs a measured **115 MB per hour** (16 kHz PCM_16 wav), so VietSpeech's
    ~1,100 h alone is ~127 GB and all three together are ~258 GB.

    Returns ``{"train": path, "val": path, "test": path}``. Cache-aware: a
    complete cache at the current ``CORPUS_VERSION`` is reused untouched.
    viVoice is never built here -- it reuses ``data_prep``'s cache, which the
    caller must have built.
    """
    import random

    import soundfile as sf
    from datasets import Dataset
    from tqdm.auto import tqdm

    import bench
    import data_prep
    import mixture

    corpus = CORPORA[name]
    paths = split_paths(name, root)
    base = corpus_dir(name, root)

    if corpus.repo is None:                        # prebuilt (viVoice)
        missing = [s for s in SPLITS
                   if not os.path.exists(os.path.join(paths[s], "dataset_info.json"))]
        if missing:
            raise FileNotFoundError(
                f"{name} expects a prebuilt cache at {base} but {missing} are "
                "missing. Build it with:\n"
                "    import data_prep; data_prep.prepare_dataset(target_hours=8.0)")
        print(f"[corpora] {name}: reusing prebuilt cache at {base}")
        return paths

    on_disk = all(os.path.exists(os.path.join(p, "dataset_info.json"))
                  for p in paths.values())
    have, have_hours = cached_stamp(base)
    want_hours = hours_tag(target_hours)
    if on_disk and have == CORPUS_VERSION and have_hours == want_hours:
        print(f"[corpora] {name}: cache hit at {base} "
              f"(v{CORPUS_VERSION}, {want_hours} h).")
        return paths
    if on_disk:
        why = (f"is v{have}, need v{CORPUS_VERSION}" if have != CORPUS_VERSION else
               f"holds {have_hours} h, asked for {want_hours} h")
        print(f"[corpora] {name}: cache at {base} {why} — rebuilding.")

    token = token or os.environ.get("HF_TOKEN")
    rng = random.Random(seed)
    audio_dir = os.path.join(base, "audio")

    # A rebuild writes fresh filenames, so anything already here is orphaned the
    # moment the new split metadata is saved -- and at whole-corpus scale that
    # is hundreds of GB of files nothing references. Clear it first.
    #
    # rmtree, so be sure of the target: corpora with a `cached_dir` (viVoice's
    # shared data/vi_asr) returned above and never reach this line, and `base`
    # for a streamed corpus is always <root>/<name>.
    assert corpus.cached_dir is None and base.endswith(name), (base, name)
    if os.path.isdir(audio_dir):
        import shutil

        stale = sum(len(f) for _, _, f in os.walk(audio_dir))
        print(f"[corpora] {name}: clearing {stale} stale wav(s) from a previous "
              f"build at {audio_dir}")
        shutil.rmtree(audio_dir)
    os.makedirs(audio_dir, exist_ok=True)

    records: list[dict] = []
    collected = 0.0
    dropped = 0
    budget = None if target_hours is None else target_hours * 3600.0
    # Redirected to a log file, tqdm writes a line per refresh rather than
    # repainting one. Over a multi-day whole-corpus build that is a gigabyte of
    # progress bar, so slow the refresh right down when there is no terminal.
    interval = 0.1 if sys.stderr.isatty() else 60.0
    pbar = (tqdm(unit="min", desc=f"streaming {name} (whole corpus)", mininterval=interval,
                 bar_format="{l_bar}{bar}| {n:.1f} min [{elapsed}]")
            if budget is None else
            tqdm(total=round(target_hours * 60, 1), unit="min", desc=f"streaming {name}",
                 mininterval=interval,
                 bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} min [{elapsed}<{remaining}]"))

    # Streamed, not downloaded: a 1,000 h corpus never lands on disk in full.
    extra = (corpus.channel_column,) if corpus.channel_column else ()
    for row in bench.stream_rows(corpus.repo, corpus.split, corpus.config,
                                 corpus.text_column, token, extra_columns=extra,
                                 shuffle_seed=seed if corpus.shuffle else None):
        text = mixture.normalize_train_text(str(row["text"]).strip())
        if not keep_clip(row["duration"], text):
            dropped += 1
            continue
        idx = len(records)
        shard = os.path.join(audio_dir, f"{idx // AUDIO_SHARD:05d}")
        if idx % AUDIO_SHARD == 0:
            os.makedirs(shard, exist_ok=True)
        wav = os.path.abspath(os.path.join(shard, f"{name}_{idx:07d}.wav"))
        sf.write(wav, row["array"], TARGET_SR)
        records.append({
            "audio_path": wav,
            "text": text,
            "duration": row["duration"],
            "channel": channel_of(row, corpus),
            "bucket": data_prep._bucket(row["duration"]),
            "source": name,
        })
        collected += row["duration"]
        pbar.update(row["duration"] / 60)
        # refresh=False, or this repaints on every clip and defeats mininterval
        # entirely -- which is a gigabyte of progress bar in a redirected log.
        pbar.set_postfix(clips=len(records), gb=round(collected / 3600 * 0.115, 1),
                         refresh=False)
        if budget is not None and collected >= budget:
            break
    pbar.close()

    how = "channel" if has_channel_signal(corpus) else "transcript hash"
    n_channels = len({r["channel"] for r in records})
    print(f"[corpora] {name}: kept {len(records)} clips ({collected / 3600:.2f} h), "
          f"dropped {dropped}; {n_channels} channels; splitting by {how}.")

    split = split_records(records, corpus, rng)

    # An empty split is unrecoverable later: datasets cannot load a zero-row
    # save_to_disk at all -- it raises a bare "IndexError: list index out of
    # range" from deep inside concat_tables, hours after the cause. Fail here,
    # where the reason is still visible.
    empty = [s for s in SPLITS if not split[s]]
    if empty:
        raise ValueError(
            f"{name}: split(s) {empty} came out empty — {len(records)} clips "
            f"across only {n_channels} channels is too few to split "
            f"{SPLIT_RATIOS}. Raise target_hours (currently {target_hours}; "
            f"None takes the whole corpus). Nothing was cached, so re-running "
            f"rebuilds cleanly.")
    if has_channel_signal(corpus) and n_channels < 10:
        print(f"[corpora] WARNING {name}: only {n_channels} channels — the test "
              f"split rests on {len({r['channel'] for r in split['test']})} of "
              f"them, so its WER will be dominated by those speakers. Raise "
              f"target_hours for a test set worth reporting.")

    os.makedirs(base, exist_ok=True)
    for s in SPLITS:
        Dataset.from_list(split[s]).save_to_disk(paths[s])
        print(f"[corpora] {name}/{s}: {len(split[s])} rows -> {paths[s]}")

    # Written last: a build that dies half-way leaves no stamp, so the next run
    # rebuilds rather than trusting a partial cache.
    with open(_version_path(base), "w") as fh:
        fh.write(f"{CORPUS_VERSION} {hours_tag(target_hours)}")
    return paths


def prepare_all(names, target_hours: float | None = 10.0, root: str = "data/corpora",
                token: str | None = None, seed: int = 42) -> dict:
    """Build each corpus that is reachable; report and skip the ones that aren't.

    Returns ``{name: paths}`` for the corpora that are usable. A gated corpus
    awaiting approval is skipped with its error message, so the notebooks run
    end to end on whatever is actually available.
    """
    built = {}
    for name in names:
        ok, msg = check_access(name, token)
        if not ok:
            print(f"[corpora] SKIP {name} — {msg}")
            continue
        print(f"[corpora] {name}: {msg}")
        built[name] = prepare_corpus(name, target_hours, root, token, seed)
    return built


# --- load ------------------------------------------------------------------- #
_COLUMNS = ["audio_path", "text", "duration", "channel", "bucket", "source"]


def load_corpus(name: str, root: str = "data/corpora"):
    """Load one corpus's splits, with a ``source`` column added if absent.

    viVoice's prebuilt cache predates this module and has no ``source`` column;
    adding it on load is what lets the three corpora concatenate.
    """
    from datasets import DatasetDict, load_from_disk

    out = {}
    for s in SPLITS:
        ds = load_from_disk(split_paths(name, root)[s])
        if "source" not in ds.column_names:
            ds = ds.add_column("source", [name] * len(ds))
        out[s] = ds.select_columns(_COLUMNS)
    return DatasetDict(out)


def load_mixture(names, split: str = "train", root: str = "data/corpora"):
    """Concatenate one split across corpora into a single training set."""
    from datasets import concatenate_datasets

    parts = [load_corpus(n, root)[split] for n in names]
    return concatenate_datasets(parts)


# --- evaluation ------------------------------------------------------------- #
def eval_rows(name: str, limit: int | None = None, split: str = "test",
              seed: int = 42, root: str = "data/corpora"):
    """The deterministic evaluation subset for one corpus.

    Both notebooks call this, which is the point: a before/after WER is only
    meaningful if the two runs scored the *same* clips. Subsampling shuffles
    first because the splits are stored channel-by-channel — taking the first N
    rows would score a handful of speakers rather than the split.
    """
    ds = load_corpus(name, root)[split]
    if limit is not None and len(ds) > limit:
        ds = ds.shuffle(seed=seed).select(range(limit))
    return ds


def score_corpus(model, processor, name: str, limit: int | None = None,
                 split: str = "test", batch_size: int = 8, seed: int = 42,
                 root: str = "data/corpora"):
    """Transcribe one corpus's evaluation subset and score it.

    Returns ``(metrics, frame)`` where ``metrics`` is ``vi_norm.wer_cer`` plus
    ``source``/``split``/``limit`` provenance, and ``frame`` holds the
    per-clip references and hypotheses.
    """
    import pandas as pd

    import bench
    import data_prep
    from vi_norm import wer_cer

    rows = list(eval_rows(name, limit, split, seed, root))
    arrays = [data_prep.read_audio(r["audio_path"])[0] for r in rows]
    hyps = bench.transcribe_arrays(model, processor, arrays,
                                   batch_size=batch_size, desc=f"eval {name}")
    frame = pd.DataFrame({
        "dataset": name,
        "audio_path": [r["audio_path"] for r in rows],
        "channel": [r["channel"] for r in rows],
        "bucket": [r["bucket"] for r in rows],
        "duration": [round(r["duration"], 2) for r in rows],
        "ref": [r["text"] for r in rows],
        "hyp": hyps,
    })
    metrics = {**wer_cer(frame["ref"], frame["hyp"]),
               "source": corpus_dir(name, root), "split": split, "limit": limit}
    return metrics, frame
