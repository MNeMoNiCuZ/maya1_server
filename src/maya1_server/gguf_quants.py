"""Catalog of GGUF quantizations available for the GGUF/llama.cpp backend.

Source: the file listing of the ``Mungert/maya1-GGUF`` HuggingFace repo
(<https://huggingface.co/Mungert/maya1-GGUF/tree/main>). Sizes are as published
there at time of writing and are indicative, not load-bearing.

Set ``MAYA1_GGUF_QUANT`` to any key below (just the short name, e.g. ``q4_k_m``)
to select which one is downloaded and loaded - the filename
(``maya1-<quant>.gguf``) is derived automatically. Roughly: smaller/lower on
this list = faster and less VRAM, at a bigger quality cost; larger/higher =
closer to full-precision quality, slower, more VRAM. ``q4_k_m`` (the default)
and ``q4_k_s`` are the usual sweet spot; go to ``q5_k_m``/``q6_k_m`` if you
have the VRAM to spare and want higher quality, or ``q2_k_s``/``q3_k_s`` if
you need the smallest/fastest possible footprint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GGUFQuant:
    filename: str
    size_gb: float
    note: str


# Ordered smallest/lowest-quality -> largest/highest-quality.
GGUF_QUANTS: dict[str, GGUFQuant] = {
    "q2_k_s":    GGUFQuant("maya1-q2_k_s.gguf",    1.30, "Smallest, lowest quality."),
    "q2_k_m":    GGUFQuant("maya1-q2_k_m.gguf",    1.36, "Very small, low quality."),
    "q3_k_s":    GGUFQuant("maya1-q3_k_s.gguf",    1.69, "Small, noticeable quality loss."),
    "q3_k_m":    GGUFQuant("maya1-q3_k_m.gguf",    1.75, "Small, moderate quality loss."),
    "q4_0":      GGUFQuant("maya1-q4_0.gguf",      2.11, "Legacy 4-bit, superseded by q4_k_*."),
    "q4_1":      GGUFQuant("maya1-q4_1.gguf",      2.10, "Legacy 4-bit, superseded by q4_k_*."),
    "q4_k_s":    GGUFQuant("maya1-q4_k_s.gguf",    2.02, "Slightly smaller/faster than q4_k_m, a bit lower quality."),
    "q4_k_m":    GGUFQuant("maya1-q4_k_m.gguf",    2.19, "Default: good speed/quality/VRAM balance."),
    "q5_k_m":    GGUFQuant("maya1-q5_k_m.gguf",    2.52, "Higher quality, more VRAM."),
    "q6_k_m":    GGUFQuant("maya1-q6_k_m.gguf",    2.83, "Close to full precision, more VRAM."),
    "q8_0":      GGUFQuant("maya1-q8_0.gguf",      3.52, "Near-lossless, largest common quant."),
    "f16_q8_0":  GGUFQuant("maya1-f16_q8_0.gguf",  4.96, "Highest quality available, largest/slowest."),
}


def resolve_filename(quant: str) -> str:
    """Return the GGUF filename for ``quant``, raising a clear error if unknown."""
    key = quant.strip().lower()
    try:
        return GGUF_QUANTS[key].filename
    except KeyError:
        valid = ", ".join(GGUF_QUANTS.keys())
        raise ValueError(
            f"Unknown MAYA1_GGUF_QUANT '{quant}'. Valid values: {valid}"
        ) from None
