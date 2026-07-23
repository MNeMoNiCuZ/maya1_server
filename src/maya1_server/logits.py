"""Logits processor restricting generation to valid Maya1 audio tokens.

For the `transformers` backend only (`engine.py`). After the SOS token,
Maya1 is only ever supposed to emit SNAC audio-code tokens
(`constants.SNAC_MIN_ID..SNAC_MAX_ID`) or the end-of-speech token
(`constants.CODE_END_TOKEN_ID`) - never stray text/header tokens. Masking
everything else out at the logits level (rather than just filtering them out
of the token stream afterwards, as `audio.extract_snac_codes` does) keeps the
model from ever "spending" a generation step on a token that has to be thrown
away, which measurably reduces garbled/truncated output on full-precision
loads (bf16/fp16/nf4).

Do NOT use this with the llama.cpp/GGUF backend (`gguf_engine.py`). GGUF
sampling there does not go through `transformers` `LogitsProcessor` hooks at
all, and forcing this kind of external logit masking through llama.cpp's own
sampling pipeline breaks generation outright (produces garbled/silent audio)
rather than constraining it cleanly - llama.cpp's own vocabulary/sampling
internals don't compose with it the way `transformers` generation does. The
GGUF backend relies solely on `audio.extract_snac_codes` to drop any stray
tokens after the fact, which is sufficient there.
"""

from __future__ import annotations

from . import constants


class SnacTokenLogitsProcessor:
    """Mask all logits except the valid SNAC-code range and end-of-speech."""

    def __init__(self, min_id: int = constants.SNAC_MIN_ID,
                 max_id: int = constants.SNAC_MAX_ID,
                 eos_id: int = constants.CODE_END_TOKEN_ID) -> None:
        self.min_id = min_id
        self.max_id = max_id
        self.eos_id = eos_id
        self._mask = None
        self._mask_shape = None

    def __call__(self, input_ids, scores):
        import torch

        if self._mask is None or self._mask_shape != scores.shape:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.min_id : self.max_id + 1] = 0.0
            mask[:, self.eos_id] = 0.0
            self._mask = mask
            self._mask_shape = scores.shape
        return scores + self._mask
