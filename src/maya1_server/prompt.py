"""Prompt construction for Maya1.

Maya1 expects a very specific framing of the input built from special tokens:

    <SOH> <BOS> <description="..."> <text> <EOT> <EOH> <SOA> <SOS>

The model then generates SNAC audio tokens until it emits the end-of-speech
token. This module builds that exact string; getting it wrong yields silence or
noise. The logic mirrors the reference ``build_prompt`` from the model card.
"""

from __future__ import annotations

from . import constants


def format_text(description: str, text: str) -> str:
    """Compose the human-readable ``<description="..."> text`` fragment."""
    description = description.strip()
    text = text.strip()
    return f'<description="{description}"> {text}'


def build_prompt(tokenizer, description: str, text: str) -> str:
    """Build the fully framed prompt string for the model.

    ``tokenizer`` is the Maya1 ``AutoTokenizer``; it is used to decode the
    special-token IDs back into the exact literal strings the model expects.
    """
    soh_token = tokenizer.decode([constants.SOH_ID])
    eoh_token = tokenizer.decode([constants.EOH_ID])
    soa_token = tokenizer.decode([constants.SOA_ID])
    sos_token = tokenizer.decode([constants.CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([constants.TEXT_EOT_ID])
    bos_token = tokenizer.bos_token

    formatted_text = format_text(description, text)

    return (
        soh_token
        + bos_token
        + formatted_text
        + eot_token
        + eoh_token
        + soa_token
        + sos_token
    )
