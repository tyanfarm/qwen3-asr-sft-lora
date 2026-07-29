"""Vietnamese ASR data pipeline for Qwen3-ASR LoRA.

Streams the gated `capleaf/viVoice` dataset, concatenates consecutive
same-channel clips into a target duration distribution (~85% in 5-30s,
~15% in 30-60s), resamples to 16 kHz mono, and writes channel-disjoint
train/val/test splits to `data/vi_asr/`.

The pure-logic helpers (sample_target_duration, plan_segments,
split_by_channel) are unit-tested in tests/test_data_prep.py and need no
network or GPU. The I/O layer (prepare_dataset, load_splits) streams the
dataset and caches the result.
"""
from __future__ import annotations

import os
import random

# --------------------------------------------------------------------------- #
# Pure logic (unit-tested, no I/O)
# --------------------------------------------------------------------------- #

LONG_PROB = 0.15
SHORT_RANGE = (5.0, 30.0)
LONG_RANGE = (30.0, 60.0)
MAX_SEGMENT_S = 60.0


def sample_target_duration(rng: random.Random) -> float:
    """Sample a target segment length in seconds.

    ~85% of samples fall in [5, 30] s and ~15% in [30, 60] s.
    """
    if rng.random() < LONG_PROB:
        lo, hi = LONG_RANGE
    else:
        lo, hi = SHORT_RANGE
    return rng.uniform(lo, hi)


def plan_segments(clips: list[dict], rng: random.Random, gap_s: float = 0.3) -> list[list[int]]:
    """Group consecutive same-channel clips into segments.

    Each clip dict must have ``channel`` and ``duration`` keys and the list
    must already be in stream order. Greedily appends clips of the same
    channel until the accumulated duration reaches a freshly sampled target,
    never exceeding ``MAX_SEGMENT_S`` (plus the trailing clip that crossed
    the threshold). Segments shorter than 5 s are dropped.

    Returns a list of index-groups (each a list of positions into ``clips``).
    """
    groups: list[list[int]] = []
    i, n = 0, len(clips)
    while i < n:
        target = sample_target_duration(rng)
        chan = clips[i]["channel"]
        group: list[int] = []
        total = 0.0
        while i < n and clips[i]["channel"] == chan:
            dur = clips[i]["duration"]
            if group and total + gap_s + dur > MAX_SEGMENT_S:
                break
            total += (gap_s if group else 0.0) + dur
            group.append(i)
            i += 1
            if total >= target:
                break
        if total >= SHORT_RANGE[0]:  # only keep segments >= 5s
            groups.append(group)
    return groups


def split_by_channel(
    segments: list[dict],
    rng: random.Random,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list]:
    """Assign whole channels to train/val/test so none spans two splits."""
    channels = sorted({s["channel"] for s in segments})
    rng.shuffle(channels)
    n = len(channels)
    n_train = max(1, int(round(ratios[0] * n)))
    n_val = max(1, int(round(ratios[1] * n))) if n - n_train >= 2 else 0
    train_ch = set(channels[:n_train])
    val_ch = set(channels[n_train:n_train + n_val])
    out: dict[str, list] = {"train": [], "val": [], "test": []}
    for s in segments:
        if s["channel"] in train_ch:
            out["train"].append(s)
        elif s["channel"] in val_ch:
            out["val"].append(s)
        else:
            out["test"].append(s)
    return out


# --------------------------------------------------------------------------- #
# I/O layer (streaming + resample + cache)
# --------------------------------------------------------------------------- #

TARGET_SR = 16000
# Swap to "linhtran92/viet_bud500" if viVoice access is denied (see README).
_DATASET_ID = "capleaf/viVoice"


def _detect_cols(features) -> tuple[str, str, str | None]:
    """Return (audio_col, text_col, channel_col) from the dataset features.

    Works across viVoice (``channel``/``text``/``audio``) and the bud500
    fallback by probing well-known column names.
    """
    from datasets import Audio

    audio_col = next((k for k, v in features.items() if isinstance(v, Audio)), None)
    if audio_col is None:  # some datasets store audio without the Audio feature type
        audio_col = next((k for k in ("audio", "wav", "speech") if k in features), None)
    text_col = next(
        (k for k in ("text", "transcription", "sentence", "transcript") if k in features),
        None,
    )
    chan_col = next(
        (k for k in ("channel", "channel_id", "source", "speaker", "speaker_id") if k in features),
        None,
    )
    if audio_col is None or text_col is None:
        raise KeyError(f"Could not detect audio/text columns in {list(features)}")
    return audio_col, text_col, chan_col


def _decode_audio(entry: dict):
    """Decode an undecoded Audio entry ({'bytes'|'path'}) to (float32 mono, sr)."""
    import io

    import numpy as np
    import soundfile as sf

    src = io.BytesIO(entry["bytes"]) if entry.get("bytes") else entry["path"]
    try:
        arr, sr = sf.read(src, dtype="float32", always_2d=False)
    except Exception:  # non-libsndfile formats (e.g. mp3) -> fall back to librosa
        import librosa

        src2 = io.BytesIO(entry["bytes"]) if entry.get("bytes") else entry["path"]
        arr, sr = librosa.load(src2, sr=None, mono=True)
        arr = arr.astype(np.float32)
    if arr.ndim > 1:  # stereo -> mono
        arr = arr.mean(axis=1).astype(np.float32)
    return arr, sr


def _stream_clips(target_hours: float, token: str | None):
    """Yield clip dicts until ``target_hours`` of audio has been collected.

    Shows a tqdm bar tracking collected audio (in minutes) toward the target,
    so the download/decode phase isn't a silent wait.
    """
    from datasets import Audio, load_dataset
    from tqdm.auto import tqdm

    ds = load_dataset(_DATASET_ID, split="train", streaming=True, token=token)
    audio_col, text_col, chan_col = _detect_cols(ds.features)
    ds = ds.cast_column(audio_col, Audio(decode=False))  # decode bytes ourselves (stable)

    collected = 0.0
    fallback_chan = 0
    target_min = target_hours * 60
    pbar = tqdm(total=round(target_min, 1), unit="min", desc="streaming viVoice",
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} min [{elapsed}<{remaining}]")
    for row in ds:
        entry = row[audio_col]
        arr, sr = _decode_audio(entry)
        dur = len(arr) / sr
        if dur <= 0 or not row[text_col]:
            continue
        # Group by channel when available; otherwise bucket ~200 consecutive
        # clips into a synthetic "chunk" so concatenation still has locality.
        chan = str(row[chan_col]) if chan_col else f"chunk{fallback_chan // 200}"
        fallback_chan += 1
        yield {
            "array": arr,
            "sr": sr,
            "text": str(row[text_col]).strip(),
            "channel": chan,
            "duration": dur,
            "start": _path_start(entry.get("path") if isinstance(entry, dict) else None),
        }
        collected += dur
        pbar.update(dur / 60)
        pbar.set_postfix(clips=fallback_chan)
        if collected >= target_hours * 3600:
            break
    pbar.close()


def _path_start(path: str | None) -> float:
    """Parse the start timestamp from a viVoice filename like ``audio_502.22_504.3.wav``.

    Returns 0.0 when no timestamp can be parsed (keeps stream order as tiebreak).
    """
    if not path:
        return 0.0
    import os as _os
    import re

    m = re.search(r"_(\d+(?:\.\d+)?)_\d+(?:\.\d+)?\.\w+$", _os.path.basename(path))
    return float(m.group(1)) if m else 0.0


def _resample(arr, orig_sr: int, target_sr: int):
    """Resample a single array; prefers soxr (fast), falls back to librosa."""
    import numpy as np

    if orig_sr == target_sr:
        return arr.astype(np.float32)
    try:
        import soxr

        return soxr.resample(arr, orig_sr, target_sr).astype(np.float32)
    except Exception:
        import librosa

        return librosa.resample(arr, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def _concat_audio(arrs: list, srs: list[int], gap_s: float = 0.3):
    """Concatenate clips with a short silence gap, resampling to 16 kHz.

    viVoice clips share one sample rate, so we concatenate at the native rate
    and resample the whole segment **once** — avoiding thousands of per-clip
    resample calls (the setup cost of which dominated the build).
    """
    import numpy as np

    if len(set(srs)) == 1:  # fast path: uniform sample rate
        sr = srs[0]
        gap = np.zeros(int(gap_s * sr), dtype=np.float32)
        parts = []
        for i, arr in enumerate(arrs):
            if i > 0:
                parts.append(gap)
            parts.append(arr.astype(np.float32))
        return _resample(np.concatenate(parts), sr, TARGET_SR)

    # mixed sample rates (fallback): resample each clip, then concatenate
    out = []
    for i, (arr, sr) in enumerate(zip(arrs, srs)):
        if i > 0:
            out.append(np.zeros(int(gap_s * TARGET_SR), dtype=np.float32))
        out.append(_resample(arr, sr, TARGET_SR))
    return np.concatenate(out)


def _bucket(dur: float) -> str:
    """Duration bucket label shared by both dataset variants."""
    if dur < 5:
        return "0-5"
    if dur <= 30:
        return "5-30"
    if dur <= 60:
        return "30-60"
    return "60+"


def prepare_dataset(
    target_hours: float = 8.0,
    out_dir: str = "data/vi_asr",
    seed: int = 42,
    token: str | None = None,
    merge: bool = True,
) -> dict:
    """Stream viVoice and cache train/val/test splits as WAV files + metadata.

    ``merge=True`` (default): concatenate consecutive same-channel clips into
    5-30s / 30-60s segments. ``merge=False``: keep the raw native clips as-is
    (one example per clip, no concatenation) — use a different ``out_dir`` so
    the two variants don't collide (e.g. ``data/vi_asr_raw``).

    Cache-aware: if all three split folders already exist, returns their
    paths without re-downloading. Returns ``{"train": path, "val": path,
    "test": path}``.
    """
    from datasets import Dataset

    paths = {s: os.path.join(out_dir, s) for s in ("train", "val", "test")}
    if all(os.path.exists(os.path.join(p, "dataset_info.json")) for p in paths.values()):
        print(f"[data_prep] cache hit at {out_dir}; skipping build.")
        return paths

    token = token or os.environ.get("HF_TOKEN")
    rng = random.Random(seed)

    clips = list(_stream_clips(target_hours, token))
    print(
        f"[data_prep] streamed {len(clips)} clips "
        f"({sum(c['duration'] for c in clips) / 3600:.2f} h)."
    )

    import soundfile as sf
    from tqdm.auto import tqdm

    # Each record is written to a 16 kHz WAV file; the dataset stores only
    # lightweight paths + metadata, so no step holds gigabytes of audio in RAM.
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    records = []

    if merge:
        # viVoice interleaves channels clip-by-clip, so group by channel (and
        # order by in-source timestamp) to make consecutive same-channel runs
        # long enough to concatenate into 5-30s / 30-60s segments.
        clips.sort(key=lambda c: (c["channel"], c.get("start", 0.0)))
        print(f"[data_prep] grouped into {len({c['channel'] for c in clips})} channels.")
        groups = plan_segments(
            [{"channel": c["channel"], "duration": c["duration"]} for c in clips], rng
        )
        for idx, g in enumerate(tqdm(groups, desc="concatenating + writing wav", unit="seg")):
            audio = _concat_audio([clips[i]["array"] for i in g], [clips[i]["sr"] for i in g])
            dur = len(audio) / TARGET_SR
            wav_path = os.path.abspath(os.path.join(audio_dir, f"seg_{idx:06d}.wav"))
            sf.write(wav_path, audio, TARGET_SR)
            records.append({
                "audio_path": wav_path,
                "text": " ".join(clips[i]["text"] for i in g),
                "duration": dur,
                "channel": clips[g[0]]["channel"],
                "bucket": _bucket(dur),
            })
    else:
        # Raw variant: keep each native viVoice clip as its own example.
        print(f"[data_prep] {len({c['channel'] for c in clips})} channels (no merge).")
        for idx, c in enumerate(tqdm(clips, desc="resampling + writing wav", unit="clip")):
            audio = _resample(c["array"], c["sr"], TARGET_SR)
            dur = len(audio) / TARGET_SR
            wav_path = os.path.abspath(os.path.join(audio_dir, f"clip_{idx:06d}.wav"))
            sf.write(wav_path, audio, TARGET_SR)
            records.append({
                "audio_path": wav_path,
                "text": c["text"],
                "duration": dur,
                "channel": c["channel"],
                "bucket": _bucket(dur),
            })

    del clips  # free the streamed clip arrays (~GBs) before saving
    _dist = {b: sum(r["bucket"] == b for r in records) for b in ("0-5", "5-30", "30-60", "60+")}
    print(f"[data_prep] built {len(records)} records; bucket dist = "
          f"{ {k: v for k, v in _dist.items() if v} }.")

    split = split_by_channel(records, rng)
    # The tables are tiny now (paths + text), so save_to_disk is instant.
    for name in tqdm(["train", "val", "test"], desc="saving splits", unit="split"):
        segs = split[name]
        d = Dataset.from_list(segs)
        os.makedirs(paths[name], exist_ok=True)
        d.save_to_disk(paths[name])
        print(f"[data_prep] {name}: {len(segs)} segments -> {paths[name]}")
    return paths


def load_splits(out_dir: str = "data/vi_asr"):
    """Load the cached train/val/test splits as a DatasetDict.

    Each row has ``audio_path`` (a 16 kHz WAV); use :func:`read_audio` to load
    the samples.
    """
    from datasets import DatasetDict, load_from_disk

    return DatasetDict(
        {s: load_from_disk(os.path.join(out_dir, s)) for s in ("train", "val", "test")}
    )


def read_audio(path: str):
    """Load a segment WAV as (float32 mono array, sample_rate)."""
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    return arr, sr
