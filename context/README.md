# context/

Reference material for publishing this project's models, kept out of the code
tree because none of it is imported.

| file | purpose |
|---|---|
| `TEMPLATE_REPO_HF.md` | Model-card shape to follow — frontmatter, then ✨ intro / 🚋 Usage / 📡 Training data / 🎯 Accuracy. |
| `MERGING.md` | Folding a trained adapter back into the base model: the fp32/fp16 precision reasoning, the two verification checks, and what changes when publishing full weights instead of an adapter. |
| `LONG_AUDIO_EVAL.md` | The long-audio test case after a fine-tune — scoring the merged 5–60 s viVoice split, why a short-clip run needs it, and the leakage check that must come first. |
| [`../.claude/skills/deploying-to-huggingface/SKILL.md`](../.claude/skills/deploying-to-huggingface/SKILL.md) | **The deploy procedure.** Repo naming, license inheritance, which files to upload, and how to verify a published card. |

The deploy guide lives under `.claude/skills/` rather than here so Claude Code
loads it automatically when a deploy comes up. Read it directly for the same
content.

Published from this repo so far:

| repo | source checkpoint | date |
|---|---|---|
| [`tyanfarm/Qwen3-ASR-1.7B-34h`](https://huggingface.co/tyanfarm/Qwen3-ASR-1.7B-34h) | `checkpoints/vi_lora` | 2026-08-03 |
