"""Pydantic request/response models for the HTTP API.

The primary synthesis endpoint is intentionally close to the OpenAI
``POST /v1/audio/speech`` shape so existing clients need minimal changes, with
two Maya1-specific additions:

* ``description`` - the natural-language voice design string. (If omitted, the
  ``voice`` field is used as the description; if both are omitted the server
  default is used.)
* emotion tags are written inline in ``input`` (e.g. ``... <laugh> ...``).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GenerationOverrides(BaseModel):
    """Shared per-request generation-parameter overrides.

    All fields default to ``None``, meaning "use the server default" - except
    ``max_new_tokens``, where ``None`` means "auto-estimate from the input's
    length" (see :mod:`maya1_server.budget`). Pass an explicit value to bypass
    the estimator entirely.
    """

    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=28,
        le=8192,
        description=(
            "Hard cap on generated audio tokens. Omit to let the server "
            "auto-estimate a safe budget from input length (recommended) - "
            "set explicitly only if you need to override the estimate."
        ),
    )
    min_new_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0)


class SpeechRequest(GenerationOverrides):
    """Body for ``POST /v1/audio/speech``."""

    input: str = Field(
        ...,
        description="Text to speak. May contain inline emotion tags like <laugh>.",
        examples=["Hello! This is Maya1 <laugh> the best open voice model."],
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "Natural-language voice design, e.g. 'Female, 30s, British accent, "
            "warm timbre, calm pacing'. Falls back to `voice`, then the server "
            "default."
        ),
    )
    voice: Optional[str] = Field(
        default=None,
        description="Alias for `description` (OpenAI compatibility).",
    )
    model: Optional[str] = Field(
        default=None,
        description="Ignored; present for OpenAI client compatibility.",
    )
    response_format: Literal["wav", "pcm"] = Field(
        default="wav",
        description="Audio container. 'pcm' is raw 16-bit LE mono @ 24kHz.",
    )

    def resolved_description(self, default: str) -> str:
        return self.description or self.voice or default


class SpeechSegment(GenerationOverrides):
    """One line of a ``POST /v1/audio/speech/batch`` request.

    Each segment is generated as its own independent inference call - this is
    deliberate. Asking the model for one long clip covering many distinct
    emotions/lines in a single generation degrades badly (drift, garbling);
    short independent segments stitched together sound far better.
    """

    input: str = Field(
        ...,
        description="Text to speak for this segment. May contain inline emotion tags.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Per-segment voice override. Falls back to the batch-level description.",
    )


class SpeechBatchRequest(BaseModel):
    """Body for ``POST /v1/audio/speech/batch``.

    Synthesises each segment independently (own generation call, own
    auto-estimated token budget unless overridden), then concatenates the
    results into a single audio file with ``gap_ms`` of silence between
    segments. Use this instead of hand-building one long `input` string
    whenever you need multiple distinct lines/emotions read back-to-back.
    """

    segments: List[SpeechSegment] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Lines to synthesise and stitch together, in order.",
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "Default voice design applied to every segment that doesn't set "
            "its own `description`. Falls back to `voice`, then the server default."
        ),
    )
    voice: Optional[str] = Field(default=None, description="Alias for `description`.")
    gap_ms: float = Field(
        default=300.0,
        ge=0.0,
        le=5000.0,
        description="Silence inserted between segments, in milliseconds.",
    )
    response_format: Literal["wav", "pcm"] = Field(default="wav")

    # Shared generation overrides, used for any segment that doesn't set its own.
    max_new_tokens: Optional[int] = Field(default=None, ge=28, le=8192)
    min_new_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0)

    def resolved_description(self, default: str) -> str:
        return self.description or self.voice or default


class SegmentResult(BaseModel):
    """Per-segment metadata returned alongside a batch synthesis (as a header)."""

    duration_sec: float
    generated_tokens: int


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    model_loaded: bool
    model_repo: str
    backend: Literal["gguf", "transformers"]
    device: str
    version: str


class EmotionsResponse(BaseModel):
    emotions: list[str] = Field(description="Bare emotion names, e.g. 'laugh'.")
    tags: list[str] = Field(description="Inline tag form, e.g. '<laugh>'.")
    count: int


class InfoResponse(BaseModel):
    name: str
    version: str
    backend: Literal["gguf", "transformers"]
    model_repo: str
    gguf_repo: Optional[str] = Field(
        default=None, description="Set only when backend='gguf'."
    )
    gguf_filename: Optional[str] = Field(
        default=None, description="Set only when backend='gguf'."
    )
    snac_repo: str
    sample_rate: int
    description_format: str
    docs_url: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
