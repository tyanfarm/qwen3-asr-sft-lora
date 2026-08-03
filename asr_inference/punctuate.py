"""Punctuation + capitalization restoration for the Zipformer transcripts.

The ASR model cannot do this itself. Its BPE vocabulary is uppercase and
unpunctuated -- ``. , : ! ;`` exist as pieces only because BPE character
coverage swept them in, and the 6000 h of training transcripts contained none,
so the transducer never emits them. There is no decoding flag that changes this;
casing and punctuation have to be *restored* by a second model that reads the
text.

We use ``dragonSwing/xlm-roberta-capu``: XLM-R fine-tuned on Vietnamese
OSCAR-2109 to predict ``. , : ?`` plus capitalization, including irregular forms
like "MobiFone" and "YouTube". It expects **lowercased** input, which is
convenient -- our transcripts are uppercase, so we lowercase and hand them over.

Cost: ~1.1 GB of weights, ~13 s to load, then ~0.015 RTF on top of the ASR's
0.025. Off by default for that reason (``--punct`` on the CLI). Never use it
before scoring -- WER is measured on punctuation-stripped text, so restoring
punctuation can only add noise.

Compatibility shims
-------------------
That repo's remote code was written for transformers 4.x and has not been
updated. Three things break under 5.x, all patched in ``_load_remote_code``:

1. ``resize_token_embeddings`` now defaults to ``mean_resizing=True``, which
   computes a covariance over meta-device weights and raises.
2. ``ModelOutput`` subclasses must now carry ``@dataclass``; theirs does not.
3. Its ``from utils import ...`` resolves against ``__file__``'s realpath, so a
   symlinked Hub cache snapshot cannot find ``verb-form-vocab.txt`` -- hence
   ``local_dir=``, which materializes real files.

Shim 3 also means the model directory goes on ``sys.path``, so its generic
module names (``utils``, ``vocabulary``) occupy those slots in ``sys.modules``
for the rest of the process. Nothing in this repo imports either name.
"""
from __future__ import annotations

import dataclasses
import functools
import sys
from pathlib import Path

REPO_ID = "dragonSwing/xlm-roberta-capu"
DEFAULT_CAPU_DIR = Path(__file__).resolve().parent / "models" / "capu"


def _load_remote_code(model_dir: Path):
    """Import the repo's ``GecBERTModel``, patching the two 5.x breakages."""
    from transformers.modeling_utils import PreTrainedModel

    original = PreTrainedModel.resize_token_embeddings
    if not getattr(original, "_capu_patched", False):
        @functools.wraps(original)
        def resize(self, *args, **kwargs):
            kwargs["mean_resizing"] = False
            return original(self, *args, **kwargs)

        resize._capu_patched = True
        PreTrainedModel.resize_token_embeddings = resize

    path = str(model_dir)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        import modeling_seq2labels

        if not dataclasses.is_dataclass(modeling_seq2labels.Seq2LabelsOutput):
            modeling_seq2labels.Seq2LabelsOutput = dataclasses.dataclass(
                modeling_seq2labels.Seq2LabelsOutput)
        from gec_model import GecBERTModel

        return GecBERTModel
    finally:
        if inserted:
            sys.path.remove(path)


def download(dest: Path = DEFAULT_CAPU_DIR) -> Path:
    """Materialize the repo (real files, not symlinks -- see shim 3 above)."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO_ID, local_dir=str(dest))
    return dest


class Punctuator:
    """Restores ``. , : ?`` and capitalization on uppercase ASR output."""

    def __init__(self, model_dir: Path | str = DEFAULT_CAPU_DIR, *,
                 auto_download: bool = True) -> None:
        model_dir = Path(model_dir)
        if not (model_dir / "modeling_seq2labels.py").exists():
            if not auto_download:
                raise FileNotFoundError(
                    f"{model_dir} not found -- run "
                    f"`python -c 'from asr_inference.punctuate import download; download()'`")
            download(model_dir)

        GecBERTModel = _load_remote_code(model_dir)
        self.model = GecBERTModel(vocab_path=str(model_dir / "vocabulary"),
                                  model_paths=str(model_dir), split_chunk=True)

    def restore(self, text: str) -> str:
        """One transcript in, punctuated and cased text out."""
        if not text.strip():
            return text
        # The model was trained on lowercase; our transcripts are uppercase.
        return self.model(text.lower())[0]

    def restore_all(self, texts: list[str]) -> list[str]:
        """Per-item restoration, so a segment never borrows its neighbour's
        sentence boundary. Slower than one big call, but SRT cues need to stand
        alone."""
        return [self.restore(t) for t in texts]
