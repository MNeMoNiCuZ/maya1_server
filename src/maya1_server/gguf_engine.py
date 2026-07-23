"""Maya1 inference engine - quantized GGUF backend, run via ``llama.cpp``.

This is the **default** backend (``MAYA1_USE_GGUF=true``): it runs the Maya1
language model as a quantized GGUF (default ``Mungert/maya1-GGUF`` /
``maya1-q4_k_m.gguf``) through ``llama-cpp-python``, instead of full-precision
``transformers``. Much faster and far lower VRAM, at a small quality cost from
quantization. The original full-precision backend (:mod:`maya1_server.engine`)
is kept fully intact and selectable via ``MAYA1_USE_GGUF=false``.

Why llama.cpp and not a plain GGUF-reader library
--------------------------------------------------
Maya1's "text" isn't really text once past the header - the model emits raw
SNAC audio-code token IDs that must flow untouched from sampling straight into
the SNAC decoder. Loading the GGUF through a generic/naive path that round-trips
those IDs through llama.cpp's own text tokenizer/detokenizer (as you would for
an ordinary chat GGUF) re-encodes them and corrupts the audio-code stream,
producing garbled "alien" noise instead of speech. This module avoids that
entirely: the prompt is encoded to raw token IDs with the same HF tokenizer
:mod:`maya1_server.prompt` already uses, those IDs are fed straight into
``Llama.generate()`` (the low-level integer-token sampling loop, not the
string-based ``__call__``/completion API), and the raw sampled IDs come back
out and go straight into the exact same SNAC decode path as the transformers
backend (:mod:`maya1_server.audio`) - never touching text in between.

Do NOT add a ``transformers``-style logits processor here. Unlike the
full-precision backend (see :mod:`maya1_server.logits`), llama.cpp's sampling
pipeline does not compose with that kind of external logit masking - it breaks
generation outright rather than constraining it. This backend relies solely on
:func:`maya1_server.audio.extract_snac_codes` to drop any stray tokens after
the fact, which is sufficient here.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from . import audio, constants, emotions
from .config import Settings
from .engine import GenerationParams, GenerationResult, _heartbeat, _is_hf_repo_cached
from .logging_setup import get_logger
from .prompt import build_prompt

logger = get_logger("maya1.gguf_engine")


class Maya1GGUFEngine:
    """Loads the GGUF model (llama.cpp) + SNAC codec and synthesises speech.

    Mirrors :class:`maya1_server.engine.Maya1Engine`'s public surface
    (``is_loaded``, ``device``, ``load()``, ``generate()``) so the HTTP layer
    (``api/routes.py``) works unmodified regardless of which backend
    ``app.py`` selects.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._llm = None
        self._tokenizer = None
        self._snac = None
        self._device = "cpu"
        self._loaded = False

    # -- lifecycle ------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        """Load the GGUF model (llama.cpp), tokenizer and SNAC decoder.

        Imports are deferred to here so the module can be imported (e.g. for
        tests) without the heavy stack.
        """
        import torch
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from snac import SNAC
        from transformers import AutoTokenizer

        s = self._settings
        want_cuda = s.device in ("auto", "cuda")
        self._device = "cuda" if (want_cuda and torch.cuda.is_available()) else "cpu"
        if s.device == "cuda" and self._device != "cuda":
            raise RuntimeError("MAYA1_DEVICE=cuda but no CUDA device is available.")

        logger.info(
            "Loading Maya1 GGUF '%s/%s' via llama.cpp (device=%s)",
            s.gguf_repo, s.gguf_filename, self._device,
        )
        if not _is_hf_repo_cached(s.gguf_repo):
            logger.info(
                "GGUF weights not found in local cache - downloading from "
                "HuggingFace now (first run only, may take a few minutes)..."
            )
        t0 = time.perf_counter()
        model_path = hf_hub_download(repo_id=s.gguf_repo, filename=s.gguf_filename)

        n_gpu_layers = s.gguf_n_gpu_layers if self._device == "cuda" else 0
        self._llm = Llama(
            model_path=model_path,
            n_ctx=s.gguf_n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=s.gguf_n_threads,
            logits_all=False,
            verbose=False,
            # llama.cpp's repeat_penalty only looks back this many tokens by
            # default (64) - far shorter than the cyclic patterns SNAC audio
            # tokens can fall into, so a stuck loop stops being "recent"
            # enough to penalize and repeats forever until max_new_tokens.
            # Widen the window to the full context. NOTE: this must be an
            # explicit positive count - llama.cpp clamps a negative
            # penalty_last_n to 0, which *disables* the repetition penalty
            # outright (the opposite of the intent) and lets every generation
            # loop until max_new_tokens.
            last_n_tokens_size=s.gguf_n_ctx,
        )
        logger.info("GGUF model loaded in %.1fs (n_gpu_layers=%d, n_ctx=%d)",
                     time.perf_counter() - t0, n_gpu_layers, s.gguf_n_ctx)

        # Tokenizer used ONLY to build/encode the prompt (see prompt.py) -
        # no model weights are loaded by this. Token IDs must match 1:1 with
        # the vocabulary the GGUF was converted from, which standard GGUF
        # conversion preserves id-for-id from the source HF tokenizer.
        self._tokenizer = AutoTokenizer.from_pretrained(s.model_repo, trust_remote_code=True)

        logger.info("Loading SNAC audio decoder '%s'", s.snac_repo)
        if not _is_hf_repo_cached(s.snac_repo):
            logger.info(
                "SNAC weights not found in local cache - downloading from "
                "HuggingFace now (first run only)..."
            )
        t0 = time.perf_counter()
        snac = SNAC.from_pretrained(s.snac_repo).eval()
        if self._device == "cuda":
            snac = snac.to("cuda")
        self._snac = snac
        logger.info("SNAC decoder loaded in %.1fs", time.perf_counter() - t0)

        self._loaded = True

        if s.warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Run one tiny generation so the first real request is not cold."""
        logger.info("Warming up engine (GGUF backend) ...")
        try:
            result = self.generate(
                description=self._settings.default_description,
                text="Hello.",
                params=GenerationParams.from_settings(self._settings),
            )
            logger.info("Warmup complete (%.2fs audio)", result.duration_sec)
        except Exception:  # pragma: no cover - warmup is best-effort
            logger.exception("Warmup failed (continuing anyway)")

    # -- inference ------------------------------------------------------------
    def generate(
        self,
        description: str,
        text: str,
        params: Optional[GenerationParams] = None,
    ) -> GenerationResult:
        """Synthesise speech for ``text`` in the voice given by ``description``.

        Thread-safe: calls are serialised (llama.cpp context state is not
        safe to share across concurrent generations either). Raises
        ``RuntimeError`` if the engine is not loaded or too few audio tokens
        were produced.
        """
        if not self._loaded:
            raise RuntimeError("Engine is not loaded; call load() first.")

        params = params or GenerationParams.from_settings(self._settings)

        if self._settings.warn_unknown_tags:
            unknown = emotions.find_unknown_tags(text)
            if unknown:
                logger.warning("Request uses unknown emotion tag(s): %s",
                               ", ".join(unknown))

        import torch

        with self._lock:
            t0 = time.perf_counter()

            prompt = build_prompt(self._tokenizer, description, text)
            input_ids = self._tokenizer(prompt, return_tensors=None)["input_ids"]
            input_len = len(input_ids)

            logger.info(
                "generate[gguf]: text_len=%d input_tokens=%d max_new=%d temp=%.2f",
                len(text), input_len, params.max_new_tokens, params.temperature,
            )

            stop_heartbeat = threading.Event()
            heartbeat = threading.Thread(
                target=_heartbeat, args=(stop_heartbeat, params.max_new_tokens), daemon=True
            )
            heartbeat.start()

            generated_ids: list = []
            try:
                for token_id in self._llm.generate(
                    input_ids,
                    top_k=0,       # disabled - transformers backend has no top-k cap either
                    top_p=params.top_p,
                    min_p=0.0,     # disabled - not part of the reference sampling recipe
                    temp=params.temperature,
                    repeat_penalty=params.repetition_penalty,
                    reset=True,
                ):
                    generated_ids.append(token_id)
                    if (
                        token_id == constants.CODE_END_TOKEN_ID
                        and len(generated_ids) >= params.min_new_tokens
                    ):
                        break
                    if len(generated_ids) >= params.max_new_tokens:
                        break
            finally:
                stop_heartbeat.set()
                heartbeat.join(timeout=1.0)

            waveform = self._decode(generated_ids, torch)

            elapsed = time.perf_counter() - t0
            duration = len(waveform) / constants.SAMPLE_RATE
            logger.info(
                "generate[gguf] done: %d tokens -> %.2fs audio in %.2fs (rtf=%.2f)",
                len(generated_ids), duration, elapsed,
                elapsed / duration if duration else 0.0,
            )

            return GenerationResult(
                audio=waveform,
                sample_rate=constants.SAMPLE_RATE,
                duration_sec=duration,
                generated_tokens=len(generated_ids),
                snac_frames=len(waveform) // 512 if len(waveform) else 0,
                elapsed_sec=elapsed,
            )

    def _decode(self, generated_ids: list, torch) -> np.ndarray:
        """Turn generated token IDs into a trimmed float32 waveform.

        Identical to :meth:`maya1_server.engine.Maya1Engine._decode` - same
        SNAC codec, same unpacking logic, deliberately not shared as a mixin
        to keep each backend's file self-contained and independently readable.
        """
        snac_tokens = audio.extract_snac_codes(generated_ids)
        if len(snac_tokens) < constants.SNAC_TOKENS_PER_FRAME:
            raise RuntimeError(
                f"Model produced too few audio tokens ({len(snac_tokens)}); "
                "cannot decode. Try a longer input or higher max_new_tokens."
            )

        levels = audio.unpack_snac_from_7(snac_tokens)
        codes = [
            torch.tensor(level, dtype=torch.long, device=self._device).unsqueeze(0)
            for level in levels
        ]

        with torch.inference_mode():
            z_q = self._snac.quantizer.from_codes(codes)
            waveform = self._snac.decoder(z_q)[0, 0].cpu().float().numpy()

        return audio.trim_warmup(waveform)
