"""Automatic ``max_new_tokens`` estimation.

Maya1 emits :data:`~maya1_server.constants.SNAC_TOKENS_PER_FRAME` tokens per
SNAC audio frame, and each frame decodes to a fixed number of samples. That
gives a fairly stable tokens-per-second-of-audio rate, which lets us estimate
how many tokens a given input is *likely* to need from its character count
alone, and generate with the model to reach a real ``EOS`` on. That way the
model can't be cut off mid-word purely because the caller forgot to raise a
token limit that has nothing to do with their text length.

This is a heuristic ceiling, not a duration guarantee: it's driven by rough
speaking-rate assumptions (see :class:`~maya1_server.config.Settings` for the
tunables), and expressive pauses/emotion tags slow real speech down
unevenly. When a caller passes ``max_new_tokens`` explicitly in the request,
that value is used as-is and this module is not consulted.
"""

from __future__ import annotations

from . import constants


def tokens_per_second_of_audio() -> float:
    """SNAC tokens consumed per second of decoded audio, from model constants."""
    samples_per_frame = 512  # SNAC decoder hop size (fixed by the codec).
    seconds_per_frame = samples_per_frame / constants.SAMPLE_RATE
    return constants.SNAC_TOKENS_PER_FRAME / seconds_per_frame


def estimate_max_new_tokens(
    text: str,
    *,
    chars_per_second: float,
    headroom: float,
    floor_tokens: int,
    ceiling_tokens: int,
) -> int:
    """Estimate a safe ``max_new_tokens`` budget for ``text``.

    Args:
        text: The text about to be spoken (tags included; they cost tokens too).
        chars_per_second: Assumed speaking rate. Lower = more generous estimate.
        headroom: Safety multiplier applied on top of the raw estimate, to
            absorb slower-than-assumed delivery (emphasis, pauses, singing).
        floor_tokens: Never estimate below this, even for very short text.
        ceiling_tokens: Never estimate above this, regardless of text length.
    """
    char_count = max(len(text), 1)
    estimated_seconds = char_count / max(chars_per_second, 0.1)
    estimated_tokens = estimated_seconds * tokens_per_second_of_audio() * headroom
    return int(min(max(estimated_tokens, floor_tokens), ceiling_tokens))
