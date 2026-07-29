# Qwen3-ASR Vietnamese LoRA

Evaluate and LoRA fine-tune [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
on Vietnamese audio from the [`capleaf/viVoice`](https://huggingface.co/datasets/capleaf/viVoice) dataset.

## What's here

| File | Purpose |
|------|---------|
| `data_prep.py` | Streams viVoice, concatenates consecutive same-channel clips into a target duration mix (~85% in 5–30 s, ~15% in 30–60 s), resamples to 16 kHz, and writes channel-disjoint `train/val/test` splits to `data/vi_asr/`. |
| `01_eval_baseline.ipynb` | Baseline WER/CER of the stock model on the test split, per duration bucket. |
| `02_lora_finetune.ipynb` | LoRA fine-tune (decoder attention + MLP) on `train`, validate on `val`, and report **before/after WER** on `test`. |
| `vi_norm.py` | Vietnamese text normalization + `wer_cer`, shared by both notebooks and `show_results.py`. |
| `bench.py` | Registry + streaming loader for the external benchmarks (VIVOS, Common Voice 17 `vi`, VLSP 2020) evaluated in section 8 of notebook 1, plus the shared batched-transcription helper. |
| `mixture.py` | Builds the multi-source **training** mixture at `data/vi_mix/` (viVoice + VIVOS + Common Voice + VLSP 2020) and the hash-assigned held-out slices. |
| `eval_lora.py` | Scores a saved adapter on any cached split, in the notebooks' output format. Used for the 1250-clip raw viVoice split, which notebook 2 does not evaluate. |
| `show_results.py` | Pretty-prints the saved metrics and the worst/best predictions, LoRA beside baseline. |
| `tests/test_data_prep.py` | Unit tests for the duration bucketing and channel-disjoint splitting (no GPU/network). |
| `tests/test_vi_norm.py` | Unit tests for the number normalization and WER metric (no GPU/network). |
| `tests/test_bench.py` | Unit tests for the benchmark registry and batching helpers (no GPU/network). |
| `tests/test_mixture.py` | Unit tests for the mixture registry, hash splitting, transcript casing and cache versioning (no GPU/network). |

### Scoring note — numbers

References write numbers as digits (`334%`) while the model transcribes what was
spoken (`ba trăm ba mươi bốn phần trăm`). Scored naively that is one substitution
plus a run of insertions, so a single formatting mismatch can cost 100% WER on a
short utterance. `vi_norm.normalize_vi` therefore expands digits to their spoken
form on **both** sides (the model sometimes emits digits too) and folds the valid
alternative readings — `tư`/`bốn`, `mốt`/`một`, `lăm`/`năm`, `ngàn`/`nghìn`,
`linh`/`lẻ` — onto one variant.

The reverse direction (spoken → digits) is unsafe in Vietnamese: `năm` means both
*five* and *year*, so `bốn mươi năm` ("forty years") would collapse to 45.

`wer_cer` reports both `wer`/`cer` and `wer_legacy`/`cer_legacy` (the metric
without number expansion) so older runs stay comparable.

### External benchmarks (notebook 1, section 8)

`bench.py` streams three published Vietnamese sets so the baseline can sit next
to numbers reported for other models:

| key | source | split | n |
|---|---|---|---|
| `vivos` | `htdung167/vivos-preprocessed-v2` | test | 760 |
| `cmv_vi` | `fixie-ai/common_voice_17_0` (`vi`) | test | 1274 |
| `vlsp2020_100h` | `data/vi_mix/heldout_vlsp2020_100h` (built by `mixture.py`) | 5% held-out slice, by transcript hash | 276 |

Mirrors are used because `datasets>=4` no longer runs loading scripts, which the
canonical `AILAB-VNUHCM/vivos` and `mozilla-foundation/common_voice_17_0` repos
both rely on.

Two caveats when comparing against PhoWhisper's published table:

1. PhoWhisper normalizes text its own way and doesn't fully specify how, so its
   numbers are a **reference point, not a like-for-like baseline**.
2. **VLSP 2020 carries no reference number, by choice.** PhoWhisper's Task-1 and
   Task-2 test sets aren't public, so we don't chase them. The VinBigData 100h
   release is used as a corpus in its own right: the mixture trains on 90% of it
   and scores the 5% held-out slice. Its transcripts are ~96% accurate by the
   publisher's estimate, so read the before/after **delta** on this set rather
   than its absolute value.

### Training mixture (notebook 2)

`mixture.py` caches ~34 h at `data/vi_mix/` in the same record schema
`data_prep` writes, so `load_splits`, `read_audio` and the collator are unchanged:

| source | training slice | ~hours |
|---|---|---|
| viVoice | reuses `data/vi_asr` train (not re-streamed) | 6.4 h |
| VIVOS | official `train` | 15 h |
| CMV-Vi | CV17 `vi` train | 3 h |
| VLSP 2020 | 90% of an 11 h stream, by transcript hash | ~10 h |

VLSP is capped hard: the full corpus is ~100 h and would otherwise be 80% of the
mixture.

**Keeping eval sets clean.** VIVOS and Common Voice have official test splits,
so training uses only their `train` rows. VLSP has no splits, so rows are
assigned by `md5(transcript)` into 90/5/5 — hashing the *text* means an
identical sentence can never land in both training and evaluation, and the
assignment is reproducible without depending on dataset iteration order. The
held-out slice is written to `data/vi_mix/heldout_vlsp2020_100h` and is what
`bench.py` scores.

**After fine-tuning, the three external benchmarks are no longer zero-shot** —
the mixture trains on their training data. The genuinely held-out,
speaker-disjoint number is the viVoice `test` result in notebook 2 section 7.

## Setup

```bash
# 1. Torch first (Blackwell / RTX 5080, CUDA 12.8):
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# 2. Everything else:
pip install -r requirements.txt
# 3. HuggingFace token (viVoice is gated — request access on the dataset page):
cp .env.example .env      # then edit .env and paste your HF_TOKEN
```

Then run `01_eval_baseline.ipynb` first (it builds the shared data cache and the
baseline metrics), then `02_lora_finetune.ipynb`.

## Environment notes (learned during setup)

- **`transformers>=5.14`** is required — the `qwen3_asr` architecture was added
  after 5.5.x. Earlier versions raise `KeyError: 'qwen3_asr'`.
- **Audio is passed to the processor as a numpy array, not a file path.** The
  processor's file loader uses torchcodec+FFmpeg, and FFmpeg 4 (Ubuntu 22.04)
  hits a "0 channels" bug. Passing decoded 16 kHz arrays avoids it entirely.
- Audio is stored as a plain struct (not a `datasets` `Audio` feature) to sidestep
  an `AudioEncoder.to_file_like` bug in `datasets` 5.0.
- Verified on RTX 5080 16 GB: inference ~4 GB, LoRA training **7.09 GB** peak
  (batch size 1 · grad-accum 16 · gradient checkpointing · bf16, attention + MLP).

## Fitting 16 GB

Training uses batch size 1 + gradient accumulation + gradient checkpointing + bf16,
LoRA (r=16) on the language-model decoder — attention and MLP, 196 projections,
17.4 M trainable params (0.85%) — with the audio encoder frozen. Measured peak on
an RTX 5080 is 7.09 GB with clips up to ~57 s (6.99 GB on the attention-only run):
adding the MLP grows the optimizer state by ~11 M params (~130 MB in bf16 + Adam
moments), not the activation memory that dominates the peak.

## Scaling the dataset — what to change in `TrainingArguments`

The notebook's `per_device_train_batch_size=1` + `gradient_accumulation_steps=16`
is tuned for a ~34 h mixture, where a 1-epoch run is a few hours. It does not
scale: measured on an RTX 5080, batch size 1 runs at **1.42 examples/s**, which is
**115 h for one epoch of a 1000 h corpus**.

Throughput at batch 1 is 1.34 ex/s on 6.3 s clips and 1.33 ex/s on 22.5 s clips —
identical. Audio length does not matter, so the GPU is idle on per-step overhead
rather than computing. Raising the *real* batch is the entire fix:

| `per_device_train_batch_size` | ex/s | peak VRAM | 1000 h = 1 epoch | speedup |
|---:|---:|---:|---:|---:|
| 1 (notebook default) | 1.42 | 4.78 GB | 114.8 h | 1.00× |
| 4 | 5.10 | 6.00 GB | 31.9 h | 3.60× |
| **8** | 9.90 | 7.59 GB | **16.4 h** | **6.99×** |
| 16 | 12.22 | 10.78 GB | 13.3 h | 8.63× |

Gradient accumulation is **not** a batch-size knob — it is a memory workaround
that fakes a large batch by running N small forward/backward passes serially, at
N× the per-step overhead. It costs nothing in wall clock and buys nothing either:

| `gradient_accumulation_steps` (batch 8) | effective batch | ex/s | 1000 h/epoch | steps/epoch |
|---:|---:|---:|---:|---:|
| 1 | 8 | 9.82 | 16.5 h | 73,106 |
| 4 | 32 | 9.82 | 16.6 h | 18,276 |
| 16 | 128 | 10.06 | 16.1 h | 4,569 |

So the rule is: **push `per_device_train_batch_size` as high as VRAM allows, then
set `gradient_accumulation_steps` to reach the effective batch you want.**
Accumulation is the remainder, not the target. Note the accumulation multiplier
goes *down* as the dataset grows — not because the effective batch shrinks (1×16 =
16 becomes 8×4 = 32, which is larger) but because the per-device batch absorbed it.

```python
per_device_train_batch_size=8,       # 1 -> 8      the only speed knob (7x)
group_by_length=True,                 # NEW        mandatory once batch > 1
length_column_name="duration",
gradient_accumulation_steps=4,        # 16 -> 4     effective batch 32
learning_rate=1.5e-4,                 # 1e-4 -> sqrt-scaled for 2x effective batch
max_steps=...,                        # replaces num_train_epochs for a partial pass
warmup_steps=1000,                    # warmup_ratio is deprecated in transformers 5
eval_steps=2000, save_steps=2000,     # 400 would be 45 evals = 2.3 h of pure eval
save_total_limit=4,                   # 2 -> 4      resume safety on a multi-day run
tf32=True,                            # free on Blackwell
gradient_checkpointing=True,          # keep it: batch 8 needs the memory back
```

**`group_by_length` is not optional above batch 1.** Clips span 0.4–80.6 s, so
batching an 80 s clip with a 0.4 s one pads the short one 200× and spends the new
throughput computing zeros. The table above was measured on length-bucketed 5–8 s
clips — it is what bucketing buys, not what you get without it. Bucketing does
concentrate the longest clips into the final batches, so tighten `MAX_TRAIN_S`
from 60 to ~30 (the mixture's p99 is 32.7 s, so ~1% of rows are lost) or those
last batches become an OOM cliff.

**Keep gradient checkpointing on.** Disabling it is a measured 2.1× at batch 1
(1.33 → 2.80 ex/s, 6.9 → 12.5 GB on an 80 s clip), but raising the batch gives 7×
and needs that memory back.

**`dataloader_num_workers=2` is already enough.** The on-the-fly collator does
66.7 examples/s per worker, so two workers deliver ~133 ex/s against the ~10 ex/s
batch-8 training consumes. Data loading is not the bottleneck at any of these
sizes.

Two things outside `TrainingArguments` matter more at 1000 h. `r=16` (17.4 M
params) will underfit ~585,000 examples — go to `r=32`/`r=64`. And the mixture
imbalance below gets worse, not better: VIVOS at 59% of rows drove both artifacts
this project has hit, and scaling the corpus scales them too.

> Caveats: these are single-run micro-benchmarks with `r=16` on length-bucketed
> 5–8 s clips, not full training runs — real throughput is lower once mixed
> lengths, eval and checkpoint I/O are included. The batch-16 row was measured on
> 5–8 s clips and will OOM on 60–80 s ones. Row counts assume the current 6.2 s
> mean clip length holds.

## One transcript style across sources

Sources disagree on transcript conventions and the model fits whatever it is
shown. VIVOS ships ALL CAPS; viVoice, Common Voice and VLSP do not. The first
fine-tune left that alone, and with VIVOS at 59% of the mixture the model learned
to condition casing on *acoustic domain* — every one of the 760 VIVOS test clips
came back uppercase, and VIVOS WER went 7.19% → 11.64%.

That is a tokenizer effect, not a formatting one. `vi_norm` lowercases before
scoring, so casing is free at the metric level; but uppercase Vietnamese has no
whole-word tokens in the Qwen vocabulary (`"VÀ NẾU BẠN"` → 8 tokens vs 3 for
`"và nếu bạn"`), so generation runs near character granularity in a token space
the base model has little language-model prior over. The resulting errors are
phonetic — `"trót"` → `"TÓT"`, `"rối loạn co bóp"` → `"ĐÓI LỌN CÓ BỚP"`.

`mixture.normalize_train_text` folds all-caps rows on ingest and leaves ordinary
case alone, so real proper-noun capitals survive. `mixture.MIX_VERSION` stamps
the cache; bumping it forces `data/vi_mix/` to rebuild instead of silently
retraining on the transcripts the bump exists to fix.

Punctuation still differs between sources (viVoice punctuates, VIVOS and VLSP do
not). That is left alone deliberately: `vi_norm` strips punctuation before scoring
*and* those tokens are in-distribution, so it costs some capacity but does not
break token-space coherence the way casing did.

`data_prep.MAX_SEGMENT_S` only caps the viVoice segments it builds — the streamed
mixture sources are uncapped, and VLSP ships a few clips of 61–81 s. Notebook 2
cell 3 therefore filters the loaded mixture at `MAX_TRAIN_S = 60.0` (4 rows of
19,885). Lower that constant if you OOM; it filters the cache in place, so no
rebuild is needed.

## Swapping the dataset

If viVoice access is denied, set `_DATASET_ID = "linhtran92/viet_bud500"` in
`data_prep.py` — the rest of the pipeline is dataset-agnostic (channels fall back
to synthetic `chunk` groups when no channel column is present).

## Security

`.env` (your `HF_TOKEN`) and the generated `data/`, `checkpoints/`, `results/`
directories are gitignored. Never commit your token.
