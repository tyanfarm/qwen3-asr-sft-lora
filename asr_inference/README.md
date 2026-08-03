# asr_inference — Vietnamese Zipformer-30M RNN-T on CPU

CPU inference for [`hynt/Zipformer-30M-RNNT-6000h`](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h)
(icefall `zipformer2` transducer, 6000 h Vietnamese, 1st place VLSP 2025), with
VAD-based chunking so hour-long recordings work.

The Hub repo ships the sherpa-onnx export — three ONNX graphs plus a 2000-piece
BPE vocabulary — so this package drives them with `onnxruntime` directly. No k2,
no sherpa-onnx wheel, nothing to build.

## Setup

```bash
pip install onnxruntime silero-vad          # torch/torchaudio/soundfile already in requirements.txt
python -m asr_inference.download            # ~130 MB into asr_inference/models/ (gitignored)
python -m asr_inference.download --punct    # + the 1.1 GB punctuation model
```

`--only int8` / `--only fp32` downloads just one precision.

## CLI

```bash
python -m asr_inference.transcribe audio.wav                       # plain text
python -m asr_inference.transcribe lecture.mp3 --format srt -o out.srt
python -m asr_inference.transcribe interview.wav --punct           # cased + punctuated
python -m asr_inference.transcribe clip.wav --int8 --threads 8 --lower
python -m asr_inference.transcribe a.wav b.wav --format json
```

| flag | default | notes |
| --- | --- | --- |
| `--int8` | off | quantized graphs; ~8% faster, +0.08 WER (measured below) |
| `--threads` | 4 | onnxruntime intra-op threads |
| `--max-seconds` | 25 | ceiling per decoded chunk; VAD picks the cut points |
| `--batch-size` | 4 | chunks per encoder pass — raise to 8-16 for long files |
| `--no-vad` | off | fall back to energy splitting (no `silero-vad` needed) |
| `--format` | `text` | `text` / `segments` / `srt` / `json` |
| `--punct` | off | restore punctuation + capitalization (see below) |
| `--lower` | off | the model emits uppercase; lowercase it for scoring |

## Python

```python
from asr_inference import ZipformerRNNT

model = ZipformerRNNT(int8=False, num_threads=4)

result = model.transcribe("meeting.wav", max_seconds=25, batch_size=8)
print(result.text)                          # whole transcript
for seg in result.segments:                 # timestamped chunks
    print(seg.start, seg.end, seg.text)
print(result.rtf)                           # real-time factor

# Benchmark corpora: one encoder pass over several short clips.
# Pads to the longest item, so group by similar duration.
texts = model.transcribe_batch([clip_a, clip_b], sample_rate=16_000)

# Optional: punctuation + capitalization (see below).
from asr_inference.punctuate import Punctuator

punct = Punctuator()
for seg in result.segments:
    seg.text = punct.restore(seg.text)
```

`transcribe` accepts a path or a float32 array in `[-1, 1]`; arrays that are not
16 kHz need `sample_rate=`.

## How long audio is handled

The encoder is **non-streaming** — attention cost grows with the square of the
input, so a 1-hour file is not slow-but-fine, it is tens of GB of activations.
`vad.py` therefore:

1. runs Silero VAD (2 MB ONNX, ~1% RTF) to find speech spans, with
   `max_speech_duration_s` set so Silero itself breaks an over-long run at its
   quietest interior point;
2. merges neighbouring spans into chunks up to `max_seconds`, only across pauses
   shorter than `max_gap` — so chunk boundaries land in silence, not mid-word;
3. pads each chunk by 0.2 s, because VAD trims onsets and a clipped first
   phoneme reliably costs the first word;
4. decodes chunks in duration-sorted batches, so a 3 s chunk is never padded out
   to sit beside a 25 s one.

Without `silero-vad` installed it falls back to `librosa` energy splitting,
which is worse on noisy audio — a fan is "loud enough" to look like speech.

## Measured on this repo's `data/vi_asr/test` (114 utts, 2453 s)

| | WER | CER | RTF |
| --- | --- | --- | --- |
| fp32 | **5.22%** | 2.94% | 0.025 |
| int8 | 5.30% | 2.97% | 0.022 |

Scored through `vi_norm.normalize_vi`. 4 threads, WSL2. A synthetic 1-hour file
transcribes in 88 s (187 VAD chunks, ~2.9 GB peak RSS, dominated by the decoded
waveform and the torch import rather than the encoder).

## Punctuation and capitalization

The ASR model emits **uppercase, unpunctuated** Vietnamese — `XIN CHÀO`, not
`Xin chào.` — and **there is no decoding flag that changes this**. Its BPE
vocabulary is uppercase; `. , : ! ;` exist as pieces only because BPE character
coverage swept them in, and the 6000 h of training transcripts contained none,
so the transducer never predicts them. Casing and punctuation have to be
*restored* afterwards by a model that reads the text.

`--punct` does that with [`dragonSwing/xlm-roberta-capu`](https://huggingface.co/dragonSwing/xlm-roberta-capu)
(XLM-R fine-tuned on Vietnamese OSCAR-2109; restores `. , : ?` plus
capitalization, including irregular forms like *MobiFone*). Restoration is per
VAD segment, so an SRT cue never borrows its neighbour's sentence boundary.

```
plain    NGAO GÁC GÌ MÀ HÉO QUEO HẾT RỒI AI MUA VINH ĐÃ VÀO BÊN TRONG SÂN KHẤU RỒI
--punct  Ngao gác gì mà héo queo hết rồi, ai mua Vinh đã vào bên trong sân khấu rồi,
```

Cost: 1.1 GB of weights, ~13 s to load, then ~0.015 RTF on top of the ASR —
10 minutes of audio goes from 17 s to 26 s of compute. Off by default.

For **scoring**, do not use it: pass `--lower`, or run `vi_norm.normalize_vi`,
which strips punctuation on both sides anyway.

Its remote code predates transformers 5.x and needs three compatibility shims;
they live in `punctuate.py` with an explanation of each.

## The other backend: this repo's LoRA adapters

`--backend qwen` runs Qwen3-ASR-1.7B with one of the adapters under
`checkpoints/`, on GPU, through the same VAD chunking and output formats — so
the two models are comparable with one command:

```bash
python -m asr_inference.transcribe audio.wav                              # zipformer, CPU
python -m asr_inference.transcribe audio.wav --backend qwen               # checkpoints/vi_lora
python -m asr_inference.transcribe audio.wav --backend qwen --adapter checkpoints/vi_lora_3ds
python -m asr_inference.transcribe audio.wav --backend qwen --adapter none  # un-tuned base
```

`--adapter` takes a **run directory** and resolves the adapter itself, printing
what it picked:

- top-level `adapter_config.json` present → use it. That is `save_pretrained`
  output, and with `load_best_model_at_end` it is the *best* checkpoint, which
  is not always the last one.
- only `checkpoint-N/` subdirectories (run still going, or killed) → highest N.
- a `checkpoint-N` path passed directly is used verbatim.

Qwen3-ASR emits cased, punctuated text natively, so `--punct` is rejected for
this backend. It is ~25x slower than the Zipformer (RTF ~0.6 on an RTX 5080 vs
0.025 on 4 CPU threads) and needs the 1.7B base model in VRAM.

Generation is delegated to `bench.transcribe_arrays` — the same function
`eval_lora.py` and both notebooks use — so a CLI transcript and a scored one
cannot drift apart. That import means the repo root has to be on the path, i.e.
run this from the project directory.

For **scoring** an adapter, use `eval_lora.py`, not this CLI: it loads the
cached split, applies `vi_norm`, and writes predictions plus metrics.

## Files

- `zipformer.py` — feature extraction, ONNX sessions, batched RNN-T greedy search
- `qwen_lora.py` — Qwen3-ASR + LoRA adapter backend
- `vad.py` — Silero VAD segmentation, chunk merging, energy fallback
- `punctuate.py` — optional punctuation/capitalization restoration
- `results.py` — `Result` / `TimedText`, shared by both backends
- `transcribe.py` — CLI
- `download.py` — Hub fetch

## Implementation notes

Feature extraction has to match icefall's training config exactly or the output
is fluent-sounding garbage: 16 kHz mono, samples in `[-1, 1]` (**not** int16
range), 80 Mel bins, 25/10 ms, `low_freq=20`, `high_freq=-400`, `dither=0`,
`snip_edges=False` — lhotse's `FbankConfig` defaults.

Decoding is icefall's `greedy_search_batch`: at most one symbol per frame, and
the decoder is re-run only for the rows that emitted, so the per-frame cost is
close to the joiner alone.

`config.json` in the Hub repo is misleadingly named — it is icefall's
`tokens.txt` (one `<piece> <id>` per line). `bpe.model` is only needed to
tokenize text yourself; decoding does not use it.
