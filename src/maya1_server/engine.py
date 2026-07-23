"""Maya1 inference engine (HuggingFace ``transformers`` backend).

This is the reference backend: it loads the 3B Maya1 model and the SNAC codec
with ``transformers`` + ``torch`` and generates audio synchronously. It is the
simplest thing that runs correctly on a single 16GB+ GPU.

Concurrency note
----------------
A single model on a single GPU cannot safely run ``generate`` from multiple
threads at once, so :meth:`Maya1Engine.generate` is guarded by a lock. Requests
are therefore serialised per process. To serve concurrent traffic, run multiple
replicas behind a load balancer, or swap this backend for the vLLM engine (see
``docs/DEPLOYMENT.md`` - "Scaling with vLLM").
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import audio, constants, emotions
from .config import Settings
from .logging_setup import get_logger
from .logits import SnacTokenLogitsProcessor
from .prompt import build_prompt

logger = get_logger("maya1.engine")


def _heartbeat(stop_event: threading.Event, max_new_tokens: int, interval: float = 5.0) -> None:
    """Log a progress line every ``interval`` seconds until ``stop_event`` is set.

    Generation is a single blocking call into ``transformers`` with no
    per-token callback we can hook cheaply, so this is the only signal we can
    give that the process is alive and working rather than hung - both for
    ``docker compose logs`` and for anyone tailing them.
    """
    t0 = time.perf_counter()
    while not stop_event.wait(interval):
        logger.info(
            "... still generating (%.0fs elapsed, budget up to %d tokens)",
            time.perf_counter() - t0, max_new_tokens,
        )


def _is_hf_repo_cached(repo_id: str) -> bool:
    """Best-effort check for whether ``repo_id`` is already in the local HF cache.

    Used only to decide whether to log a "downloading now" heads-up before a
    slow first-run fetch; a wrong answer here has no functional effect.
    """
    try:
        from huggingface_hub import scan_cache_dir
        return any(repo.repo_id == repo_id for repo in scan_cache_dir().repos)
    except Exception:
        return False


@dataclass
class GenerationParams:
    """Per-request generation parameters, defaulted from :class:`Settings`."""

    max_new_tokens: int
    min_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float

    @classmethod
    def from_settings(cls, s: Settings) -> "GenerationParams":
        return cls(
            max_new_tokens=s.max_new_tokens,
            min_new_tokens=s.min_new_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            repetition_penalty=s.repetition_penalty,
        )


@dataclass
class GenerationResult:
    """Result of a single synthesis call."""

    audio: np.ndarray          # float32 waveform in [-1, 1] at 24 kHz
    sample_rate: int
    duration_sec: float
    generated_tokens: int
    snac_frames: int
    elapsed_sec: float


class Maya1Engine:
    """Loads the model + codec and synthesises speech."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._snac = None
        self._device = "cpu"
        self._loaded = False
        self._snac_logits_processor = SnacTokenLogitsProcessor()

    # -- lifecycle ------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        """Load the model, tokenizer and SNAC decoder into memory.

        Imports of ``torch``/``transformers``/``snac`` are deferred to here so
        the module can be imported (e.g. for tests) without the heavy stack.
        """
        import torch
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self._settings
        self._device = self._resolve_device(torch)
        torch_dtype = self._resolve_dtype(torch)

        logger.info("Loading Maya1 model '%s' (device=%s, dtype=%s)",
                    s.model_repo, self._device, s.dtype)
        if not _is_hf_repo_cached(s.model_repo):
            logger.info(
                "Model not found in local cache - downloading from HuggingFace "
                "now (first run only, several GB, may take a few minutes)..."
            )
        t0 = time.perf_counter()
        self._model = AutoModelForCausalLM.from_pretrained(
            s.model_repo,
            torch_dtype=torch_dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )
        if self._device != "cuda":
            self._model = self._model.to(self._device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            s.model_repo, trust_remote_code=True
        )
        logger.info("Model loaded in %.1fs (%d vocab tokens)",
                    time.perf_counter() - t0, len(self._tokenizer))

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
        logger.info("Warming up engine ...")
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

        Thread-safe: calls are serialised. Raises ``RuntimeError`` if the model
        is not loaded or if too few audio tokens were produced.
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
        from transformers import LogitsProcessorList

        with self._lock:
            t0 = time.perf_counter()

            prompt = build_prompt(self._tokenizer, description, text)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]
            if self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            logger.info(
                "generate: text_len=%d input_tokens=%d max_new=%d temp=%.2f",
                len(text), input_len, params.max_new_tokens, params.temperature,
            )

            stop_heartbeat = threading.Event()
            heartbeat = threading.Thread(
                target=_heartbeat, args=(stop_heartbeat, params.max_new_tokens), daemon=True
            )
            heartbeat.start()
            try:
                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=params.max_new_tokens,
                        min_new_tokens=params.min_new_tokens,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        repetition_penalty=params.repetition_penalty,
                        do_sample=True,
                        eos_token_id=constants.CODE_END_TOKEN_ID,
                        pad_token_id=self._tokenizer.pad_token_id,
                        # Restricts every generated step to valid SNAC-code
                        # tokens + EOS. transformers-backend only - see
                        # logits.py's module docstring for why this must NOT
                        # be ported to the llama.cpp/GGUF backend.
                        logits_processor=LogitsProcessorList([self._snac_logits_processor]),
                    )
            finally:
                stop_heartbeat.set()
                heartbeat.join(timeout=1.0)

            generated_ids = outputs[0, input_len:].tolist()
            waveform = self._decode(generated_ids, torch)

            elapsed = time.perf_counter() - t0
            duration = len(waveform) / constants.SAMPLE_RATE
            logger.info(
                "generate done: %d tokens -> %.2fs audio in %.2fs (rtf=%.2f)",
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
        """Turn generated token IDs into a trimmed float32 waveform."""
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

    # -- helpers --------------------------------------------------------------
    def _resolve_device(self, torch) -> str:
        want = self._settings.device
        if want == "cpu":
            return "cpu"
        if want == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("MAYA1_DEVICE=cuda but no CUDA device is available.")
            return "cuda"
        # auto
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, torch):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self._settings.dtype]
