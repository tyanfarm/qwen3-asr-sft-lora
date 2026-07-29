"""Vietnamese text normalization for WER/CER scoring.

The references in this dataset write numbers as digits ("334%", "1872") while
the model transcribes what was actually spoken ("ba trăm ba mươi bốn phần
trăm"). Scored naively that is one substitution plus a pile of insertions, so a
single formatting mismatch can cost 100% WER on a short utterance.

We therefore expand digits to their spoken form on *both* sides — the model
sometimes emits digits too — rather than parsing spoken numbers back into
digits. The reverse direction is unsafe in Vietnamese: "năm" is both *five* and
*year*, so "bốn mươi năm" (forty years) would collapse to 45.

Numbers have several valid readings (334 is "ba mươi bốn" or "ba mươi tư"), so
after expansion both sides are folded onto one canonical variant. That lets us
generate only the plain forms here and let VARIANTS absorb the alternatives.

Both notebooks and show_results.py import from here, so the metric has exactly
one definition.
"""
from __future__ import annotations

import re
import unicodedata

# --- character filter (keeps Vietnamese diacritics, drops punctuation) ------ #
_DIACRITICS = ("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
               "ùúủũụưứừửữựỳýỷỹỵđ")
_VN = "a-z0-9\\s" + _DIACRITICS
_KEEP = re.compile(f"[^{_VN}]")
_LETTER = f"[0-9a-z{_DIACRITICS}]"  # "does a word character follow?"

# --- number -> words -------------------------------------------------------- #
_ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_SCALES = ["", "nghìn", "triệu", "tỷ"]


def _three(n: int, full: bool) -> list[str]:
    """Read 0-999. ``full`` forces the hundreds slot ("không trăm hai mươi hai")."""
    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    out: list[str] = []
    if hundreds or full:
        out += [_ONES[hundreds], "trăm"]
    if tens == 0:
        if units:
            out += ["lẻ", _ONES[units]] if (hundreds or full) else [_ONES[units]]
    elif tens == 1:
        out.append("mười")
        if units:
            out.append(_ONES[units])
    else:
        out += [_ONES[tens], "mươi"]
        if units:
            out.append(_ONES[units])
    return out


def num_to_vi(n: int) -> str:
    """Spell a non-negative integer in Vietnamese, plain (unfolded) variants."""
    n = int(n)
    if n == 0:
        return "không"
    groups: list[int] = []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)
    if len(groups) > len(_SCALES):  # beyond "tỷ" — leave it to the caller
        raise ValueError("number too large to spell")
    out: list[str] = []
    for i in range(len(groups) - 1, -1, -1):
        if groups[i] == 0:
            continue
        out += _three(groups[i], full=i != len(groups) - 1)
        if _SCALES[i]:
            out.append(_SCALES[i])
    return " ".join(out)


def _spell(match: re.Match) -> str:
    try:
        return f" {num_to_vi(match.group())} "
    except ValueError:
        return match.group()


# --- digit patterns --------------------------------------------------------- #
_THOUSAND_SEP = re.compile(r"(?<=\d)\.(?=\d{3}(?!\d))")
_DECIMAL = re.compile(r"(\d+),(\d+)")
_CLOCK = re.compile(rf"(\d{{1,2}})h(\d{{1,2}})?(?!{_LETTER})")
_INTEGER = re.compile(r"\d+")

# Spoken form of unit abbreviations that follow a number. Longest first so "mm"
# wins over "m"; extend as the data demands.
_UNITS = {
    "%": "phần trăm", "kg": "ki lô gam", "km": "ki lô mét", "cm": "xăng ti mét",
    "mm": "mi li mét", "ml": "mi li lít", "gr": "gram", "g": "gram",
    "m": "mét", "l": "lít",
}
_UNIT_RE = re.compile(
    r"(\d+)\s*(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True))
    + rf")(?!{_LETTER})")

# Valid alternative readings, folded onto one form so either scores as correct.
_VARIANTS = {"tư": "bốn", "mốt": "một", "lăm": "năm", "ngàn": "nghìn",
             "linh": "lẻ"}


def _expand_numbers(t: str) -> str:
    while _THOUSAND_SEP.search(t):
        t = _THOUSAND_SEP.sub("", t)
    t = _CLOCK.sub(
        lambda m: f" {num_to_vi(m.group(1))} giờ "
                  + (num_to_vi(m.group(2)) if m.group(2) else ""), t)
    t = _DECIMAL.sub(
        lambda m: f" {num_to_vi(m.group(1))} phẩy "
                  + " ".join(_ONES[int(d)] for d in m.group(2)) + " ", t)
    t = _UNIT_RE.sub(lambda m: f" {num_to_vi(m.group(1))} {_UNITS[m.group(2)]} ", t)
    return _INTEGER.sub(_spell, t)


def normalize_vi(t, expand_numbers: bool = True) -> str:
    """Lowercase, expand numbers, strip punctuation, fold spoken variants.

    ``expand_numbers=False`` reproduces the original normalizer, so the old and
    new metrics can be reported side by side.
    """
    if not isinstance(t, str):  # empty CSV cells parse as float NaN
        t = "" if t is None or t != t else str(t)
    t = unicodedata.normalize("NFC", t).lower()
    if expand_numbers:
        t = _expand_numbers(t)
    t = re.sub(r"\s+", " ", _KEEP.sub(" ", t)).strip()
    if not expand_numbers:
        return t
    return " ".join(_VARIANTS.get(w, w) for w in t.split())


def wer_cer(refs, hyps) -> dict:
    """Corpus WER/CER, reported both with and without number expansion.

    The ``*_legacy`` keys are the pre-number-normalization metric, kept so
    older runs stay comparable.
    """
    import jiwer

    out: dict = {}
    for suffix, expand in (("", True), ("_legacy", False)):
        r = [normalize_vi(x, expand_numbers=expand) for x in refs]
        h = [normalize_vi(x, expand_numbers=expand) for x in hyps]
        pairs = [(a, b) for a, b in zip(r, h) if a]  # drop empty refs
        r, h = [a for a, _ in pairs], [b for _, b in pairs]
        out[f"wer{suffix}"] = jiwer.wer(r, h)
        out[f"cer{suffix}"] = jiwer.cer(r, h)
        out["n"] = len(r)
    return out
