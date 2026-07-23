"""Runtime configuration for the Maya1 server.

All settings are read from environment variables with the ``MAYA1_`` prefix,
falling back to the defaults below. In Docker these are supplied via the
``.env`` file / compose ``environment`` block.

Example::

    MAYA1_HOST=0.0.0.0
    MAYA1_PORT=41217
    MAYA1_DTYPE=bfloat16
    MAYA1_MAX_NEW_TOKENS=2048
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import constants
from .gguf_quants import GGUF_QUANTS, resolve_filename


class Settings(BaseSettings):
    """Server settings, populated from ``MAYA1_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MAYA1_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- HTTP server ---------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 41217
    log_level: str = "INFO"
    cors_allow_origins: str = "*"  # comma-separated list, or "*" for all

    # --- Model / codec repositories ------------------------------------------
    model_repo: str = constants.DEFAULT_MODEL_REPO
    snac_repo: str = constants.DEFAULT_SNAC_REPO

    # --- Device / precision --------------------------------------------------
    # "auto" lets the engine pick cuda when available, else cpu.
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # Warm the model with a tiny generation at startup so the first real
    # request is not slow. Set MAYA1_WARMUP=false to skip.
    warmup: bool = True

    # --- Backend selection -----------------------------------------------------
    # Default backend: the quantized GGUF model run through llama.cpp
    # (`gguf_engine.py`) - much faster and far lower VRAM than the full-precision
    # `transformers` backend, at a small quality cost from quantization. Set
    # MAYA1_USE_GGUF=false to fall back to the original `transformers` backend
    # (`engine.py`), kept fully intact and selectable.
    use_gguf: bool = True
    gguf_repo: str = "Mungert/maya1-GGUF"
    # Which quantization to download/load - just the short name (e.g. "q4_k_m").
    # See `gguf_quants.GGUF_QUANTS` for the full catalog (also listed in
    # docs/DEPLOYMENT.md §4a); invalid values fail fast at startup with the
    # valid list in the error message.
    gguf_quant: str = "q4_k_m"
    # Advanced override: an exact filename in gguf_repo, bypassing gguf_quant
    # entirely (e.g. to load a custom/third-party GGUF build). Leave unset to
    # use gguf_quant.
    gguf_filename_override: Optional[str] = None
    # Context window: must fit prompt + max_new_tokens with room to spare.
    gguf_n_ctx: int = 8192
    # -1 = offload all layers to GPU (llama.cpp/GGML). Ignored (forced to 0) on CPU.
    gguf_n_gpu_layers: int = -1
    # None = let llama.cpp pick (usually os.cpu_count()).
    gguf_n_threads: Optional[int] = None

    @field_validator("gguf_quant")
    @classmethod
    def _validate_gguf_quant(cls, v: str) -> str:
        key = v.strip().lower()
        if key not in GGUF_QUANTS:
            valid = ", ".join(GGUF_QUANTS.keys())
            raise ValueError(f"MAYA1_GGUF_QUANT must be one of: {valid} (got '{v}')")
        return key

    @property
    def gguf_filename(self) -> str:
        return self.gguf_filename_override or resolve_filename(self.gguf_quant)

    # --- Default generation parameters (per the Maya1 model card) ------------
    # Hard ceiling on generation length. Auto-estimation (below) never exceeds
    # this; an explicit per-request `max_new_tokens` can still go higher, up to
    # the API's own hard limit (8192) - that's an intentional escape hatch for
    # callers who know better than the estimator.
    max_new_tokens: int = 2048
    min_new_tokens: int = 28          # >= 4 SNAC frames
    temperature: float = 0.4
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # --- Automatic max_new_tokens estimation ----------------------------------
    # When a request omits `max_new_tokens`, the server estimates a safe budget
    # from the input's character count instead of just using a flat default -
    # this is what prevents long inputs from being cut off mid-word. See
    # `maya1_server.budget` for the formula.
    auto_tokens_chars_per_second: float = 12.0   # assumed speaking rate
    auto_tokens_headroom: float = 1.6            # safety multiplier
    auto_tokens_floor: int = 256                 # never estimate below this

    # Hard cap on input text length (characters) to protect the GPU queue.
    max_input_chars: int = 2_000

    # Warn (via logs) when the caller uses an unrecognised <tag>.
    warn_unknown_tags: bool = True

    # Default voice description used when a request omits one.
    default_description: str = Field(
        default=(
            "Realistic male voice in the 30s age with american accent. "
            "Normal pitch, warm timbre, conversational pacing."
        )
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list."""
        value = self.cors_allow_origins.strip()
        if value == "*":
            return ["*"]
        return [origin.strip() for origin in value.split(",") if origin.strip()]


# Importable singleton. Instantiating reads the environment once.
settings = Settings()
