"""Unattended run: build the corpus caches, then execute notebooks 3 and 4.

This exists because the full run is measured in hours, not minutes, and a
Jupyter kernel is the wrong place to hold it. Scale, from measurements on this
box:

    disk        115 MB per audio-hour of 16 kHz PCM_16 wav
    download    300 audio-hours per wall-hour on the shard-prefetch path
    training    9.82 examples/s at batch 8

    TARGET_HOURS = 100 per corpus  ->  400 h, ~46 GB, ~1.5 h to build, 8.9 h to train
    TARGET_HOURS = None            ->  ~2,740 h, ~315 GB, ~9 h to build, ~58 h to train

Training time scales with *examples*, not hours, and the four corpora differ in
clip length by 2.9x -- 100 h is 71,238 viVoice train clips but 126,326 Bud500
ones. Four corpora at 100 h each is 315,029 training examples (measured).

Keep TARGET_HOURS equal to HOURS_PER_CORPUS in both notebooks. The caches are
shared and stamped with the hours they were built at, so a mismatch rebuilds
(costing a download) rather than silently mixing scales.

Each corpus is built separately, so a failure costs one corpus rather than the
whole set, and a completed corpus is version-stamped and skipped on re-run.
Re-running this script after any failure resumes at the first unfinished corpus.

    python run_3ds.py            # build, then run notebook 3, then notebook 4
    python run_3ds.py --build    # build only
"""
from __future__ import annotations

import os
import sys
import time
import traceback

NAMES = ["vivoice_full", "vietspeech", "vieneu", "bud500"]
NOTEBOOKS = ["03_eval_baseline_3ds.ipynb", "04_lora_finetune_3ds.ipynb"]

TARGET_HOURS = 100.0        # per corpus; None takes each corpus in full
ATTEMPTS = 3                # per corpus, on transient stream failures


def _stamp(msg: str) -> None:
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build() -> list:
    """Build every reachable corpus. Returns the names that succeeded."""
    import corpora

    done = []
    for name in NAMES:
        # A corpus already built at TARGET_HOURS needs nothing from the network,
        # so it is not probed -- a probe can only fail and lose a finished build.
        if corpora.cache_is_complete(name, TARGET_HOURS):
            _stamp(f"{name} — complete cache, nothing to build")
            done.append(name)
            continue
        ok, msg = corpora.check_access(name)
        if not ok:
            _stamp(f"SKIP {name} — {msg}")
            continue
        target = "whole corpus" if TARGET_HOURS is None else f"{TARGET_HOURS:g} h"
        t0 = time.time()
        # A long stream dies on transient HTTP faults -- an earlier whole-corpus
        # run lost 829 h of viVoice to "Cannot send a request, as the client has
        # been closed". A build discards nothing on failure (the stamp is
        # written last), so a retry is just the download again: cheap at 100 h,
        # and the alternative is losing the corpus to a blip.
        for attempt in range(1, ATTEMPTS + 1):
            _stamp(f"building {name} ({target}), attempt {attempt}/{ATTEMPTS} — {msg}")
            try:
                corpora.prepare_corpus(name, target_hours=TARGET_HOURS)
            except ValueError:                # a split came out empty: retrying
                traceback.print_exc()         # the same stream cannot fix that
                _stamp(f"FAILED {name} — not retryable; raise TARGET_HOURS")
                break
            except Exception:                 # noqa: BLE001 - one corpus must not
                traceback.print_exc()         # take down the rest of the run
                if attempt == ATTEMPTS:
                    _stamp(f"FAILED {name} after {ATTEMPTS} attempts "
                           f"({(time.time() - t0) / 3600:.1f} h) — continuing with "
                           "the others; re-run to retry this one")
                    break
                _stamp(f"{name} attempt {attempt} failed — retrying in 60 s")
                time.sleep(60)
                continue
            done.append(name)
            _stamp(f"built {name} in {(time.time() - t0) / 3600:.2f} h")
            break
    return done


def run_notebook(path: str) -> None:
    """Execute a notebook in place, so its outputs land where the user reads them."""
    import nbformat
    from nbclient import NotebookClient

    _stamp(f"executing {path}")
    t0 = time.time()
    nb = nbformat.read(path, as_version=4)
    # timeout=None: the training cell alone runs for hours.
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": os.getcwd()}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)              # keep partial output on failure
    _stamp(f"finished {path} in {(time.time() - t0) / 3600:.2f} h")


def main() -> int:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        print("Set HF_TOKEN in .env", file=sys.stderr)
        return 1

    import shutil

    # 115 MB per audio-hour, plus headroom for checkpoints and the HF cache.
    need = 0.115 * len(NAMES) * (TARGET_HOURS if TARGET_HOURS is not None else 750)
    floor = need + 25
    free = shutil.disk_usage(".").free / 1e9
    _stamp(f"start — {free:.0f} GB free (this configuration needs ~{need:.0f} GB)")
    if free < floor:
        print(f"Only {free:.0f} GB free; this build needs ~{need:.0f} GB plus room "
              f"for checkpoints. Free space or lower TARGET_HOURS.", file=sys.stderr)
        return 1

    built = build()
    _stamp(f"corpora ready: {built}")
    if not built:
        print("No corpus could be built; nothing to run.", file=sys.stderr)
        return 1

    if "--build" in sys.argv:
        return 0

    for path in NOTEBOOKS:
        try:
            run_notebook(path)
        except Exception:                     # noqa: BLE001 - report and stop
            traceback.print_exc()
            _stamp(f"FAILED {path} — outputs up to the failing cell were saved")
            return 1
    _stamp("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
