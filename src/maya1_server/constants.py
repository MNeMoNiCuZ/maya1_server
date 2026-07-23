"""Model-level constants for Maya1.

These values are defined by the ``maya-research/maya1`` model and the
``hubertsiuzdak/snac_24khz`` codec. They are NOT tunable configuration - they
describe the fixed token layout the model was trained with. Changing them will
produce garbage audio.

Source: the Maya1 model card (Quick Start reference implementation).
"""

from __future__ import annotations

# --- Special token IDs (extended Llama tokenizer vocabulary) -----------------
BOS_ID: int = 128000              # Beginning of sequence
TEXT_EOT_ID: int = 128009         # End of text turn
CODE_START_TOKEN_ID: int = 128257  # Start-of-speech (SOS)
CODE_END_TOKEN_ID: int = 128258    # End-of-speech (EOS) - generation stop token
SOH_ID: int = 128259              # Start of header
EOH_ID: int = 128260              # End of header
SOA_ID: int = 128261              # Start of audio
CODE_TOKEN_OFFSET: int = 128266   # SNAC codes are offset by this value

# --- SNAC audio-token layout -------------------------------------------------
SNAC_MIN_ID: int = 128266         # First valid SNAC code token
SNAC_MAX_ID: int = 156937         # Last valid SNAC code token
SNAC_TOKENS_PER_FRAME: int = 7    # 7 flat tokens unpack to 3 hierarchical levels
SNAC_CODEBOOK_SIZE: int = 4096    # Codes per level are taken modulo this value

# --- Audio output ------------------------------------------------------------
SAMPLE_RATE: int = 24_000         # 24 kHz mono, as produced by the SNAC decoder
NUM_CHANNELS: int = 1
SAMPLE_WIDTH_BYTES: int = 2       # 16-bit PCM on the wire
WARMUP_SAMPLES: int = 2_048       # Leading samples trimmed after decode

# Default HuggingFace repositories.
DEFAULT_MODEL_REPO: str = "maya-research/maya1"
DEFAULT_SNAC_REPO: str = "hubertsiuzdak/snac_24khz"
