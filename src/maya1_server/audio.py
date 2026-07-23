"""SNAC token decoding and audio encoding.

The model emits a flat stream of token IDs. Turning that into a playable file is
a three-step process:

1. :func:`extract_snac_codes`  - keep only the audio-code tokens (drop the EOS
   marker and any stray text tokens).
2. :func:`unpack_snac_from_7`  - regroup the flat 7-token frames into the three
   hierarchical codebook levels the SNAC decoder expects.
3. the SNAC decoder (in :mod:`maya1_server.engine`) turns codes into a float32
   waveform, which :func:`encode_wav` / :func:`float_to_pcm16` serialise.

All logic here is pure/numpy and has no torch or model dependency, so it is
cheap to unit-test.
"""

from __future__ import annotations

import io
import wave
from typing import List

import numpy as np

from . import constants


def extract_snac_codes(token_ids: List[int]) -> List[int]:
    """Return only the SNAC code tokens from a generated token stream.

    Everything up to the end-of-speech token is scanned; tokens outside the
    valid SNAC range (text tokens, headers, etc.) are dropped.
    """
    try:
        eos_idx = token_ids.index(constants.CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)

    return [
        token_id
        for token_id in token_ids[:eos_idx]
        if constants.SNAC_MIN_ID <= token_id <= constants.SNAC_MAX_ID
    ]


def unpack_snac_from_7(snac_tokens: List[int]) -> List[List[int]]:
    """Unpack flat 7-token SNAC frames into 3 hierarchical levels.

    Each frame of 7 tokens maps to::

        L1: 1 code   (slot 0)
        L2: 2 codes  (slots 1, 4)
        L3: 4 codes  (slots 2, 3, 5, 6)

    Codes are de-offset by ``CODE_TOKEN_OFFSET`` and taken modulo the codebook
    size. Returns ``[l1, l2, l3]``.
    """
    if snac_tokens and snac_tokens[-1] == constants.CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]

    per_frame = constants.SNAC_TOKENS_PER_FRAME
    frames = len(snac_tokens) // per_frame
    snac_tokens = snac_tokens[: frames * per_frame]

    if frames == 0:
        return [[], [], []]

    offset = constants.CODE_TOKEN_OFFSET
    codebook = constants.SNAC_CODEBOOK_SIZE

    l1: List[int] = []
    l2: List[int] = []
    l3: List[int] = []

    for i in range(frames):
        slots = snac_tokens[i * per_frame : (i + 1) * per_frame]
        l1.append((slots[0] - offset) % codebook)
        l2.extend(
            [
                (slots[1] - offset) % codebook,
                (slots[4] - offset) % codebook,
            ]
        )
        l3.extend(
            [
                (slots[2] - offset) % codebook,
                (slots[3] - offset) % codebook,
                (slots[5] - offset) % codebook,
                (slots[6] - offset) % codebook,
            ]
        )

    return [l1, l2, l3]


def concatenate_waveforms(
    waveforms: List[np.ndarray], gap_ms: float, sample_rate: int
) -> np.ndarray:
    """Join ``waveforms`` in order, inserting ``gap_ms`` of silence between each.

    Used by the multi-segment batch endpoint to stitch several independently
    generated clips (e.g. one per emotion) into a single continuous waveform,
    so callers never have to parse/merge WAV bytes themselves.
    """
    if not waveforms:
        return np.zeros(0, dtype=np.float32)
    if len(waveforms) == 1:
        return waveforms[0]

    gap_samples = int(round(sample_rate * gap_ms / 1000.0))
    silence = np.zeros(gap_samples, dtype=np.float32)

    parts: List[np.ndarray] = []
    for i, waveform in enumerate(waveforms):
        if i > 0:
            parts.append(silence)
        parts.append(waveform)
    return np.concatenate(parts)


def trim_warmup(audio: np.ndarray) -> np.ndarray:
    """Drop the leading SNAC warmup samples (decoder ramp-up artefacts)."""
    if len(audio) > constants.WARMUP_SAMPLES:
        return audio[constants.WARMUP_SAMPLES :]
    return audio


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float32 waveform in [-1, 1] to little-endian 16-bit PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def encode_wav(audio: np.ndarray) -> bytes:
    """Encode a float32 waveform as a 24 kHz mono 16-bit WAV file (in memory)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(constants.NUM_CHANNELS)
        wav.setsampwidth(constants.SAMPLE_WIDTH_BYTES)
        wav.setframerate(constants.SAMPLE_RATE)
        wav.writeframes(float_to_pcm16(audio))
    return buffer.getvalue()


def wav_header(num_samples: int) -> bytes:
    """Return a 44-byte WAV header for a mono 16-bit stream of ``num_samples``.

    Used for streaming responses where the body is emitted as raw PCM after the
    header. When the final length is unknown, pass a large placeholder; most
    players tolerate a header whose declared size exceeds the delivered bytes.
    """
    data_bytes = num_samples * constants.SAMPLE_WIDTH_BYTES * constants.NUM_CHANNELS
    byte_rate = (
        constants.SAMPLE_RATE * constants.NUM_CHANNELS * constants.SAMPLE_WIDTH_BYTES
    )
    block_align = constants.NUM_CHANNELS * constants.SAMPLE_WIDTH_BYTES

    header = io.BytesIO()
    header.write(b"RIFF")
    header.write((36 + data_bytes).to_bytes(4, "little"))
    header.write(b"WAVE")
    header.write(b"fmt ")
    header.write((16).to_bytes(4, "little"))          # PCM fmt chunk size
    header.write((1).to_bytes(2, "little"))           # audio format = PCM
    header.write(constants.NUM_CHANNELS.to_bytes(2, "little"))
    header.write(constants.SAMPLE_RATE.to_bytes(4, "little"))
    header.write(byte_rate.to_bytes(4, "little"))
    header.write(block_align.to_bytes(2, "little"))
    header.write((constants.SAMPLE_WIDTH_BYTES * 8).to_bytes(2, "little"))
    header.write(b"data")
    header.write(data_bytes.to_bytes(4, "little"))
    return header.getvalue()


# MIME type per supported response format.
CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/L16; rate=24000; channels=1",
}
