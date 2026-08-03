# Long-audio test case (merged viVoice), after a fine-tune

## Why this test exists

The 300 h run in `04_lora_finetune_3ds.ipynb` trains on **short clips only**.
Measured on the sources it draws from:

| corpus | what the source ships | mean clip |
|---|---|---|
| `vivoice_full` | native viVoice clips, unmerged | 3.87 s |
| `vietspeech` | social-media utterances | ~2–6 s |
| `vieneu` | TTS read speech | short (measure after the build) |

So the adapter never sees a 30 s utterance. The base model did — Qwen3-ASR has no
architectural length ceiling (1 s conv chunks, 8 s attention windows, 13 audio
tokens per second, cost linear in duration), and it was pretrained on long-form
material. A LoRA on the decoder's attention **and** MLP is exactly the surface
that could learn "utterances end after ~4 seconds" and start truncating, drifting,
or looping on a 45 s input.

**None of the three per-corpus WERs in notebook 4 can see that.** They all score
short clips. This test is the one that can.

The material for it is already on disk: `data/vi_asr`, the merged cache
`data_prep.py` built for notebooks 1–2, where consecutive same-channel viVoice
clips are concatenated into 5–60 s segments. It survived the `data/corpora` clear.

## What is on disk (measured, not estimated)

`data/vi_asr` — merged, 932 MB:

| split | segments | hours | mean | max | channels | 5–30 s | 30–60 s |
|---|---|---|---|---|---|---|---|
| train | 980 | 6.41 | 23.5 s | 59.5 s | 140 | 772 | 208 |
| val | 208 | 1.35 | 23.4 s | 58.4 s | 18 | 162 | 46 |
| **test** | **114** | **0.68** | **21.5 s** | **57.9 s** | **17** | **94** | **20** |

`data/vi_asr_raw` — the same builder with `merge=False`, so the *unmerged* clips:

| split | clips | hours | mean | max | channels | 0–5 s | 5–30 s |
|---|---|---|---|---|---|---|---|
| test | 1250 | 1.44 | 4.1 s | 17.5 s | 18 | 892 | 358 |

Both are reachable through `corpora.py`: the `"vivoice"` entry has
`cached_dir="data/vi_asr"`, and `load_corpus` adds the missing `source` column on
load, so `corpora.score_corpus(model, processor, "vivoice", ...)` works unchanged.

## Two checks before any number here means anything

### 1 · The merged test split is probably **not** held out

`vivoice_full` streams `capleaf/viVoice` from the head of the train split with
`shuffle=False` (`corpora.CORPORA["vivoice_full"]` sets no `shuffle`, so
`stream_rows` gets `shuffle_seed=None`). `data_prep._stream_clips` streams the
same repo, same split, from the same head. The 8.4 h that built `data/vi_asr` is
therefore almost certainly **inside** the 100 h that `vivoice_full` trains on —
same channels, same underlying audio, merely re-segmented.

Run this once `data/corpora/vivoice_full` exists:

```python
from datasets import load_from_disk
import corpora

train_ch = set(corpora.load_corpus("vivoice_full")["train"]["channel"])
test_ch = set(load_from_disk("data/vi_asr/test")["channel"])
print(len(test_ch), "long-form test channels;", len(test_ch & train_ch), "seen in training")
```

- **Overlap 0** — genuinely held out. Report it beside notebook 4's other numbers.
- **Overlap > 0** (expected) — this is an **in-domain** long-form probe, not a
  held-out one. It still answers the question that matters ("did training on short
  clips break long-form?"), because a model that has *seen this speaker's audio*
  and still regresses on long segments has regressed for length reasons, not
  novelty reasons. But label it in-domain everywhere it is reported. An unlabelled
  delta here reads as a generalization claim it cannot support.

To get a clean held-out version, restrict to channels not in `train_ch` and check
what is left — at 17 test channels there may not be enough to bootstrap.

### 2 · Confirm the baseline scored these exact clips

`results/baseline_metrics.json` already holds a stock-model score on this split
from notebook 1, so the base model does not need re-running — *if* the split has
not been rebuilt since. `data/vi_asr/test` has a later mtime than that JSON, so
verify rather than assume:

```python
import pandas as pd
from datasets import load_from_disk

old = set(pd.read_csv("results/baseline_predictions.csv")["audio_path"])
new = set(load_from_disk("data/vi_asr/test")["audio_path"])
print(len(old), len(new), "identical" if old == new else "REBUILT — re-run the base")
```

If they differ, score the base model on the current split before comparing
anything. A before/after across two different clip sets is not a delta.

## Running it

`eval_lora.py` already does exactly this job, including the per-bucket breakdown
that is the whole point:

```bash
python eval_lora.py --data data/vi_asr --split test \
                    --adapter checkpoints/vi_lora_3ds \
                    --out results/3ds_longform \
                    --batch-size 2
```

Writes `results/3ds_longform_predictions.csv` and `_metrics.json` with `overall`
plus `by_bucket` for `5-30` and `30-60`.

**`--batch-size 2`, not the default 8.** Notebook 4 already drops
`per_device_eval_batch_size` to 4 because 8 OOMs on 60 s clips, and that is
*training* eval on the short corpora. `bench.transcribe_arrays` sorts by length
before batching, so the longest segments land in a batch together — the worst
case, not the average. `corpora.MAX_CLIP_S`'s comment records a measured 6.7 GB
peak at ~57 s on this 16 GB box, single clip.

For the matched short-clip control on the same corpus:

```bash
python eval_lora.py --data data/vi_asr_raw --split test \
                    --adapter checkpoints/vi_lora_3ds --out results/3ds_raw
```

Caveat on that pairing: the merged and raw caches are **not speaker-matched** —
their test splits share 1 channel out of 17/18. `plan_segments` consumes the
shared `random.Random(42)` before `split_by_channel` runs in the merged path and
not in the raw path, so the two builds shuffle channels differently. Read the
merged-vs-raw gap as length *confounded with speaker*, not length alone. A truly
matched control means concatenating `data/vi_asr_raw/test`'s own clips per
channel, so both variants cover identical audio.

## The reference point, and what to expect

From the 34 h run (`checkpoints/vi_lora`), which trained on the merged 5–60 s
segments — the opposite length distribution to the 300 h run:

| split | bucket | n | base WER % | 34 h LoRA WER % | delta |
|---|---|---|---|---|---|
| merged | 5–30 | 94 | 5.90 | 6.08 | +0.19 |
| merged | 30–60 | 20 | 6.09 | 5.90 | −0.19 |
| merged | overall | 114 | 5.96 | 6.03 | +0.07 |
| raw | 0–5 | 892 | 4.77 | 5.13 | **+0.35** |
| raw | 5–30 | 358 | 2.91 | 2.78 | −0.13 |
| raw | overall | 1250 | 3.76 | 3.86 | +0.09 |

That run trained long and **regressed on short** (0–5 s, +0.35 pts, the largest
move in the table). The 300 h run inverts the training distribution, so the
falsifiable prediction is the mirror image: **0–5 s should improve, and 30–60 s is
where a regression would show.** If 30–60 s degrades while notebook 4's per-corpus
WERs all improve, that is the finding — and it is invisible without this test.

Note the honesty constraint that comes with it: at n=20 in the 30–60 s bucket,
nothing under roughly a full point is distinguishable from noise. Notebook 2's
run 1 is the cautionary case in this repo — a 0.02 pt "win" on viVoice that was
2 word errors in 10,155. Bootstrap it (notebook 4 section 10 has the code) or
report the bucket as directional only.

## Pitfalls, with the numbers attached

| pitfall | the number | consequence |
|---|---|---|
| `max_new_tokens=440` in `bench.transcribe_arrays` | refs run 5.19 tok/s mean, 5.84 p95 → 440 covers **~75 s** | fine at 60 s (longest ref here is 280 tokens at 48.9 s); past ~75 s the hypothesis is silently truncated and the deletions look like a model failure |
| `MAX_SEGMENT_S = 60` / `corpora.MAX_CLIP_S = 60` | hard caps at build time | anything longer needs custom concatenation; `plan_segments` will not produce it |
| eval batch size | 6.7 GB measured at ~57 s, single clip | batch 8 OOMs; use 1–2 |
| concatenation seams | 0.3 s gap, transcripts joined with `" "` | refs read as run-on sentences with no punctuation at the joins. `vi_norm` strips punctuation before scoring so WER is unaffected, but do not read the CSV as evidence about punctuation |
| punctuation/casing claims | `vi_norm` normalizes both away | measure them from the CSV directly, as `.claude/skills/deploying-to-huggingface/SKILL.md` requires — no WER in this repo can support such a claim |
| context length | text `max_position_embeddings` = 65,536; 13 audio tok/s → ~84 min | not the binding constraint at this scale. VRAM binds far sooner |

## Going past 60 s

If the 30–60 s bucket holds up and you want the real long-form question (2–10 min,
what `Qwen3-ASR-Toolkit` chunks for), `plan_segments` cannot build it —
`MAX_SEGMENT_S` stops at 60. Concatenate a channel's clips directly with
`data_prep._concat_audio`, and raise `max_new_tokens` in step with duration
(≥ 6 tok/s of audio, from the p95 above, plus margin). At that tier expect to
transcribe one clip at a time, and expect the comparison to be against a chunked
pipeline rather than a single forward pass — a 2 h file is ~93,600 audio tokens,
past the 65,536 position budget, so it *cannot* be a single request regardless of
VRAM.

## Related

- [`MERGING.md`](MERGING.md) — folding the adapter into the base for a standalone repo.
- [`../.claude/skills/deploying-to-huggingface/SKILL.md`](../.claude/skills/deploying-to-huggingface/SKILL.md) — what a published card may claim.
