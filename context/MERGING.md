# Merging the LoRA adapter into the base model

How to fold `checkpoints/vi_lora_3ds` back into `Qwen/Qwen3-ASR-1.7B-hf` and get a
standalone model directory that loads exactly like Qwen's own repo — no `peft`
dependency, no adapter plumbing.

This is a **post-training** step. Run `04_lora_finetune_3ds.ipynb` first: the
verification in section 4 compares against `results/3ds_lora_metrics.json`, and
without that file there is nothing to check the merge against.

## 1 · Decide whether to merge at all

Merging is not an upgrade — it is a trade. Both artifacts are valid outputs.

| | adapter (`checkpoints/vi_lora_3ds`) | merged |
|---|---|---|
| size | ~70 MB | ~3.4 GB (fp16) |
| loads with | `peft` + base model | `transformers` alone |
| serving | vLLM `--enable-lora` | plain `vllm serve` |
| A/B against base | free — disable the adapter | needs both repos resident |
| stack several adapters | yes | no |
| inference speed | one extra matmul per projection | marginally faster |
| provenance | self-evidently derivative | looks like a standalone model |

That last row is the one that matters for publishing, and it is covered in
section 5.

**Recommendation: keep the adapter as the primary artifact** and produce a merged
build for convenience — for handing someone a drop-in replacement, or for a
serving stack that does not speak LoRA.

## 2 · Merge

Merging is weight arithmetic: `W ← W + (B@A)·(α/r)` across the 196 adapted
projections. No activations, no gradients, no optimizer state — none of
training's memory profile applies.

**Do it on CPU.** It needs ~6.8 GB of ordinary RAM (1.7 B params × 4 bytes in
fp32), takes a couple of minutes for 196 small matmuls, and leaves the GPU free
for whatever else is running.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

BASE = "Qwen/Qwen3-ASR-1.7B-hf"
ADAPTER = "checkpoints/vi_lora_3ds"
OUT = "checkpoints/vi_lora_3ds_merged"

# fp32 on CPU: the delta is computed as a sum over r=16 terms, and fp32
# accumulation costs nothing here. See section 3 for what this does and does
# not buy.
base = AutoModelForMultimodalLM.from_pretrained(BASE, dtype=torch.float32,
                                                device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()          # returns the plain base class

model = model.to(torch.float16)           # in place — see section 3 for why fp16
model.save_pretrained(OUT, safe_serialization=True)

# The processor is not part of the adapter, so a merged directory without it is
# not self-contained: no feature extractor, no tokenizer, no chat template.
AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)
```

The result is architecturally identical to the base. Our LoRA targets only
`model.language_model` decoder attention and MLP (see the `TARGET_RE` assertions
in notebook 4 section 4), so the audio encoder comes through bit-identical and
the config is unchanged. `AutoModelForMultimodalLM.from_pretrained(OUT)` works
with no `trust_remote_code`, no custom class.

**Watch peak memory at the cast.** `model.to(torch.float16)` converts in place.
Building a second full-precision copy instead briefly holds ~10 GB.

## 3 · Why fp32 to merge, and fp16 to save

The base ships in bf16, so "convert to fp32 first" looks redundant. The reasoning
is narrower than it is usually stated, and the *save* dtype matters more than the
*merge* dtype.

**The operands are not the problem; the addition is.** `W` in bf16 is fine and
`ΔW` on its own is fine. The loss happens in `W + ΔW`, where a small quantity is
swamped by a large one. bf16 carries 8 significand bits, so its relative ULP is
`2⁻⁸ ≈ 0.39%`. At `W_ij = 0.5` the gap to the next representable value is ~0.002,
and any update below half that rounds away entirely. LoRA deltas at `lr=1e-4`
over one epoch land in exactly that band.

**But fp32 does not rescue that if you save in bf16.** `round_bf16(W + ΔW)` is
the same value whether the sum was computed in fp32 or in correctly-rounded bf16,
and PyTorch upcasts internally for bf16 addition. What fp32 genuinely buys is
narrower: accurate accumulation inside the `B@A` product, insurance against a
backend that accumulates in low precision, and no compounding if adapters are
ever merged in sequence. Cheap, worth doing, not dramatic.

**The lever that actually moves the result is the save dtype:**

| dtype | significand bits | relative ULP | bytes |
|---|---|---|---|
| bf16 | 8 | 0.39% | 2 |
| **fp16** | **11** | **0.049%** | **2** |
| fp32 | 24 | 6e-8 | 4 |

fp16 costs the same two bytes as bf16 and resolves the delta **8× finer**. bf16's
advantage is exponent range, which matters for activations and gradients during
training, not for stored weights — those sit well inside fp16's ±65504. This is
why merged models are commonly distributed in fp16.

Section 4 is what settles it either way. If merged WER matches the adapter's, the
precision was sufficient and the argument above is moot.

## 4 · Verify — two checks, both required

### Numeric: does it reproduce the adapter?

Score the merged model exactly as notebook 4 did — same `EVAL_LIMIT`, same
`SEED`, same corpora — and diff against the adapter's saved metrics. Load it in
the dtype you intend to **serve**, not the dtype you merged in, or you are
measuring a configuration nobody will run.

```python
import json
import torch
import corpora
from transformers import AutoModelForMultimodalLM, AutoProcessor

EVAL_LIMIT, SEED = 500, 42               # must match notebook 4
merged = AutoModelForMultimodalLM.from_pretrained(
    OUT, dtype=torch.float16, attn_implementation="sdpa", device_map="cuda").eval()
processor = AutoProcessor.from_pretrained(OUT)
processor.tokenizer.padding_side = "left"

adapter_metrics = json.load(open("results/3ds_lora_metrics.json"))
for name in adapter_metrics:
    m, f = corpora.score_corpus(merged, processor, name, limit=EVAL_LIMIT,
                                batch_size=8, seed=SEED)
    f.to_csv(f"results/3ds_merged_{name}_predictions.csv", index=False)
    delta = 100 * (m["wer"] - adapter_metrics[name]["wer"])
    flag = "OK" if abs(delta) < 0.05 else "INVESTIGATE"
    print(f"{name:<14} merged {100*m['wer']:6.2f}%  adapter "
          f"{100*adapter_metrics[name]['wer']:6.2f}%  delta {delta:+.3f} pts  {flag}")
```

A few hundredths of a point is rounding. Anything larger means the merge lost
something — re-run it saving in fp32 and compare again to isolate whether it is
precision or a genuine bug.

### Style: what does it actually emit?

`vi_norm` lowercases and strips punctuation before scoring, so **no WER or CER in
this repo can support a claim about casing or punctuation**. Measure it directly.

This is not hypothetical here. Notebook 4 trains on a mixture that is one-third
VietSpeech, which is lowercase and unpunctuated, and an earlier run in this
project collapsed punctuation from 96.4% → 0.0% on VIVOS while it *rose* to 99.1%
on viVoice — the model learned to predict the punctuation convention from the
acoustic domain. Invisible to WER, immediately obvious to anyone reading a
transcript.

```python
import pandas as pd
for name in adapter_metrics:
    for tag in ("lora", "merged"):
        h = pd.read_csv(f"results/3ds_{tag}_{name}_predictions.csv")["hyp"].astype(str)
        print(f"{name:<14} {tag:<7}"
              f" punct {100 * h.str.contains(r'[.,?!]').mean():5.1f}%"
              f"  lead-cap {100 * h.str.match(r'^\s*[A-ZÀ-Ỹ]').mean():5.1f}%")
```

The merged and adapter rows should agree. They are the same weights; a divergence
here means the merge changed behaviour in a way the WER check missed.

## 5 · Publishing a merged repo

Everything in `.claude/skills/deploying-to-huggingface` applies, with three
differences that come from shipping full weights instead of an adapter.

**The license chain is the blocker, and merging sharpens it.** License is
inherited from the training data, not the base model:

```python
from huggingface_hub import HfApi
api = HfApi()
api.model_info("Qwen/Qwen3-ASR-1.7B-hf").cardData.get("license")     # apache-2.0
for d in ("capleaf/viVoice", "NhutP/VietSpeech", "pnnbao-ump/VieNeu-TTS-140h"):
    print(d, api.dataset_info(d).cardData.get("license"))
```

`capleaf/viVoice` is **CC BY-NC-SA 4.0**, so the derived weights are CC BY-NC-SA
— non-commercial, share-alike — despite the Apache-2.0 base. Check the other two
before publishing; VieNeu is gated and may carry redistribution terms of its own.

An adapter is self-evidently derivative: nobody can use it without fetching the
base. A merged repo *looks* like a drop-in replacement for an Apache-2.0 model,
so downloaders will assume permissive terms unless the card says otherwise in
plain language. State the chain explicitly.

**Different frontmatter.** A merged repo is not a PEFT artifact:

```yaml
language: [vi]
license: cc-by-nc-sa-4.0            # from the data, not the base model
base_model: Qwen/Qwen3-ASR-1.7B-hf
library_name: transformers          # NOT peft
pipeline_tag: automatic-speech-recognition
tags: [merged, qwen3-asr]           # no lora/peft adapter tags
datasets: [capleaf/viVoice, NhutP/VietSpeech, pnnbao-ump/VieNeu-TTS-140h]
metrics: [wer, cer]
```

**Different upload patterns.** The adapter guidance (`allow_patterns` limited to
`adapter_config.json`, `adapter_model.safetensors`, `README.md`) does not apply —
a merged repo legitimately ships gigabytes. Still use `allow_patterns`, to keep
`checkpoint-*/` optimizer state and `training_args.bin` out:

```python
api.upload_folder(
    repo_id=REPO, folder_path=OUT, repo_type="model",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "README.md"],
    ignore_patterns=["checkpoint-*/*", "training_args.bin", "optimizer.pt"],
    commit_message="Qwen3-ASR-1.7B merged with Vietnamese LoRA (300 h)",
)
```

Name the repo from the hours `mixture.summarize` actually reported in notebook 4,
not from the 300 h this was planned at.

## 6 · Serving

A merged repo needs no LoRA support in the serving stack:

```bash
vllm serve checkpoints/vi_lora_3ds_merged --max-model-len 32768
```

versus the adapter form, which keeps the base shared and lets you A/B on one
server:

```bash
vllm serve Qwen/Qwen3-ASR-1.7B-hf --enable-lora \
    --lora-modules vi=checkpoints/vi_lora_3ds
```

Either can back `Qwen3-ASR-Toolkit` for long audio — point `--api-url` at the
server and raise `-j` well above its default of 4, which otherwise starves
continuous batching. Note the toolkit's `--max-segment-seconds 180` default comes
from DashScope's cap and is not a constraint of your own server.

## Common mistakes

| Mistake | Consequence |
|---|---|
| Merging on the GPU | Pointless contention; it is CPU weight arithmetic |
| Building an fp32 copy instead of casting in place | ~10 GB peak instead of ~7 GB |
| Saving bf16 "because the base is bf16" | 8× coarser resolution on the delta, for the same 2 bytes |
| Skipping `processor.save_pretrained` | Repo has weights but no feature extractor or tokenizer |
| Verifying with a different dtype than you serve | Measures a configuration nobody runs |
| Claiming casing/punctuation from memory | Unfalsifiable by WER, and wrong last time |
| Taking the license from the base model | Publishes NC-derived weights as Apache-2.0 |
