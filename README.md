# Maya1 TTS Server

A standards-based, Dockerised inference server for
**[Maya1](https://maya1.org/)** — the open-source (Apache 2.0) 3B-parameter
speech model for **expressive, voice-designed text-to-speech**. Describe a voice
in plain English, drop `<laugh>` / `<whisper>` / `<angry>` tags inline, and get
24 kHz audio back over a simple HTTP API.

```text
POST /v1/audio/speech
{ "description": "Female, 30s, British accent, warm timbre, calm pacing.",
  "input": "It's good to see you again <chuckle> I've missed our chats." }
        │
        ▼
   maya-research/maya1  ──►  SNAC 24 kHz codec  ──►  hello.wav
```

- **Voice design from natural language** — no fixed speaker list.
- **17 inline emotion tags** — expression placed exactly where you want it.
- **OpenAI-compatible** `/v1/audio/speech` endpoint (plus a `description` field).
- **One command to run** in Docker with GPU passthrough.
- **Standards-based**: FastAPI + Pydantic v2, OpenAPI docs, 12-factor config,
  structured logging, health probes.

> Full model details — architecture, voice-design guide, the complete emotion
> list, prompt format, generation parameters and SNAC decoding — are in
> **[docs/MODEL.md](docs/MODEL.md)**. Read it before extending the server.

---

## Installing

Two supported paths: **Docker** (recommended, isolates the whole CUDA stack
for you) or **manual / bare metal** (`python server.py` directly). Whichever
you pick, the single most common cause of "it runs, but on CPU" is a
CPU-only `torch` wheel silently installed instead of a CUDA build - see the
manual path below for how to avoid and detect that.

### Option A: Docker (recommended)

**Prerequisites:** NVIDIA GPU (**8 GB+ VRAM** with the default GGUF backend;
**16 GB+** if you switch to the full-precision `transformers` backend), recent
driver, Docker, and the
[NVIDIA Container Toolkit](docs/DEPLOYMENT.md#1-prerequisites).

```bash
cp .env.example .env          # optional tweaks
docker compose up --build     # build, download weights (~6 GB, first run only), serve
docker compose logs -f        # wait for "Startup complete; ready to serve"
```

The `Dockerfile` installs the CUDA-matched `torch` (cu128) and builds
`llama-cpp-python` with CUDA support for you - no manual torch step needed
with this path.

Then:

```bash
scripts/smoke_test.sh         # health + emotions + writes out.wav

# or a raw request:
curl -X POST http://localhost:41217/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"description":"Energetic female event host, American accent, clear diction.",
       "input":"Welcome everyone! <excited> Tonight is going to be unforgettable."}' \
  --output hello.wav
```

Alternatively on Windows: double-click `sample_generation.bat` (~5 minutes
for the demo generation). It runs `scripts/smoke_test.py` interactively and
prints the device (CPU/GPU) the server reports, plus progress while waiting.

Interactive API docs: <http://localhost:41217/docs>

### Option B: Manual / bare metal

The server is a normal Python app under `src/` and can be run directly with
`python server.py`. You still need an NVIDIA GPU + driver, and (for the
default GGUF backend) a CUDA toolkit + C++ build chain to compile
`llama-cpp-python`.

**Windows:**

```bat
venv_create.bat        REM creates venv/, offers to install requirements.txt
venv_activate.bat       REM if not already activated
python setup.py         REM copies .env, installs torch + llama-cpp-python (guided)
python server.py
```

**Linux:**

```bash
./setup.sh              REM creates venv/, activates it, runs setup.py
python server.py
```

`setup.py` is interactive and safe to re-run: it copies `.env.example` to
`.env` (only if `.env` doesn't already exist), attempts to install a
CUDA-matched torch build, installs the rest of `requirements.txt`, and then
walks you through building `llama-cpp-python` with CUDA support - printing
the exact prerequisites (CUDA Toolkit, C++ build tools) and manual fallback
commands for your OS if the automatic build fails or you'd rather do it
yourself.

#### Installing torch manually (do this if `setup.py` skipped it, or if generation is slow / GPU sits idle)

`pip install torch` on its own gives you a **CPU-only** build with no
warning - the package installs fine, imports fine, and only fails silently
at runtime (`torch.cuda.is_available()` returns `False`), which also forces
`MAYA1_GGUF_N_GPU_LAYERS` to be ignored and the GGUF backend to fall back to
CPU-only inference. This project's server gates ALL GPU usage (both the
`transformers` backend and llama.cpp's GPU layer offload for the GGUF
backend) on `torch.cuda.is_available()`, so a CPU-only torch build means
CPU-only inference everywhere, even though `MAYA1_DEVICE=auto`/`cuda` and
`MAYA1_GGUF_N_GPU_LAYERS=-1` are both set correctly in `.env`.

To install the right build **manually, matching your own Python version and
CUDA driver** (don't just copy this project's cu128 default blindly):

1. Check your NVIDIA driver's supported CUDA version: `nvidia-smi` (top-right
   of the output, e.g. `CUDA Version: 13.0`).
2. Check your Python version: `python --version` (torch wheels are built per
   Python minor version, e.g. 3.11 vs 3.12).
3. Go to <https://pytorch.org/get-started/locally/>, select your OS, pip,
   your Python version, and a CUDA version **at or below** what `nvidia-smi`
   reported (driver support is backward compatible - a cu128 wheel runs fine
   on a CUDA 13.0 driver, for example). Copy the exact command it gives you,
   e.g.:

   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu128 torch>=2.7
   ```

4. If torch was already installed as a CPU-only build (this project's
   `setup.py` reports "already installed - skipping" without checking this),
   force a clean reinstall instead of a plain `pip install`:

   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall torch>=2.7
   ```

5. Verify it actually took, before starting the server:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   `torch.__version__` should show a `+cuXXX` suffix (not `+cpu`), and
   `cuda.is_available()` must print `True`. If it doesn't, the driver, CUDA
   version, or Python version you picked in step 3 doesn't match your system
   - go back and re-check them.

See
[docs/DEPLOYMENT.md §4a](docs/DEPLOYMENT.md#4a-backends-gguf-llamacpp-vs-transformers)
for backend details, or set `MAYA1_USE_GGUF=false` in `.env` to skip
`llama-cpp-python` entirely and use the full-precision `transformers`
backend (needs 16 GB+ VRAM).

---

## HTTP API (summary)

| Method & path | Purpose |
|---|---|
| `GET /` | Service metadata. |
| `GET /health` | Readiness probe (`model_loaded` true/false). |
| `GET /v1/emotions` | List the supported inline emotion tags. |
| `POST /v1/audio/speech` | Synthesise speech → WAV or raw PCM. |

Full reference with every field and more examples: **[docs/API.md](docs/API.md)**.

---

## Emotion tags

Place any of these inline in `input`, e.g. `... that's hilarious <laugh> ...`:

`<laugh>` `<laugh_harder>` `<sigh>` `<chuckle>` `<gasp>` `<angry>` `<excited>`
`<whisper>` `<cry>` `<scream>` `<sing>` `<snort>` `<exhale>` `<gulp>` `<giggle>`
`<sarcastic>` `<curious>`

The running server serves this list at `GET /v1/emotions`. See
[docs/MODEL.md §3](docs/MODEL.md#3-emotion-tags) for usage guidance.

---

## Repository layout

```
.
├── Dockerfile                # CUDA 12.8 devel image; torch (cu128) + llama-cpp-python (CUDA) + deps
├── docker-compose.yml        # one-command GPU deployment + weight-cache volume
├── .dockerignore
├── .env.example              # all MAYA1_* settings, documented
├── requirements.txt          # Python deps (torch installed separately in image)
├── server.py                 # bare-metal entrypoint: `python server.py`
├── setup.py                  # bare-metal setup: .env, torch, deps, guided llama-cpp-python build
├── setup.sh                  # Linux: creates venv + runs setup.py
├── venv_create.bat           # Windows: creates venv + installs requirements.txt
├── README.md                 # this file
├── docs/
│   ├── MODEL.md              # model architecture, voice design, emotions, SNAC
│   ├── API.md                # full HTTP API reference
│   └── DEPLOYMENT.md         # Docker/GPU deployment, config, scaling, troubleshooting
├── scripts/
│   ├── entrypoint.sh         # container entrypoint (launches the server)
│   ├── download_model.py     # pre-fetch weights into the HF cache
│   ├── smoke_test.sh         # end-to-end check against a running server
│   └── example_client.py     # minimal stdlib Python client
└── src/
    └── maya1_server/         # the application package
        ├── __main__.py       # `python -m maya1_server`
        ├── app.py            # FastAPI factory + lifespan (loads model)
        ├── config.py         # env-driven settings (pydantic-settings)
        ├── constants.py      # fixed model token IDs + audio constants
        ├── emotions.py       # emotion-tag registry
        ├── prompt.py         # builds the model's prompt framing
        ├── audio.py          # SNAC unpack + WAV/PCM encoding (pure/numpy)
        ├── engine.py         # transformers backend (full precision, optional)
        ├── gguf_engine.py    # llama.cpp/GGUF backend (default, MAYA1_USE_GGUF=true)
        ├── logits.py         # SNAC-token logits processor (transformers backend only)
        ├── budget.py         # auto max_new_tokens estimation from input length
        ├── logging_setup.py  # structured logging
        └── api/
            ├── schemas.py    # request/response models
            └── routes.py     # endpoint handlers
```

---

## Configuration

Twelve-factor: everything is an environment variable with the `MAYA1_` prefix,
loaded from `.env`. Highlights (full list in [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `MAYA1_PORT` | `41217` | HTTP port. |
| `MAYA1_USE_GGUF` | `true` | `true` = quantized GGUF via llama.cpp (default, fast, low VRAM). `false` = full-precision `transformers` backend. See [docs/DEPLOYMENT.md §4a](docs/DEPLOYMENT.md#4a-backends-gguf-llamacpp-vs-transformers). |
| `MAYA1_DEVICE` | `auto` | `auto` / `cuda` / `cpu`. |
| `MAYA1_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` (transformers backend only). |
| `MAYA1_TEMPERATURE` | `0.4` | Default sampling temperature. |
| `MAYA1_MAX_NEW_TOKENS` | `2048` | Generation upper bound (ceiling for auto-estimation too — see [docs/API.md](docs/API.md#automatic-length-max_new_tokens)). |
| `MAYA1_AUTO_TOKENS_CHARS_PER_SECOND` | `12.0` | Assumed speaking rate for auto-estimating `max_new_tokens` from input length. |
| `MAYA1_AUTO_TOKENS_HEADROOM` | `1.6` | Safety multiplier on the auto-estimate. |
| `MAYA1_WARMUP` | `true` | Warm the model at startup. |

---

## Implementation checklist (for the agent picking this up)

This repo is a **complete, runnable reference server**, not stubs. If you are
extending or verifying it, work through this list:

- [ ] **Bring up the stack** — `docker compose up --build`; confirm `/health`
      reports `"model_loaded": true` and `scripts/smoke_test.sh` produces audible
      `out.wav`.
- [ ] **Verify the model contract** — the special token IDs in
      `src/maya1_server/constants.py` and the prompt framing in `prompt.py` must
      match the [model card](https://huggingface.co/maya-research/maya1). These
      are the two things that silently break audio if wrong.
- [ ] **Confirm the emotion list** — `src/maya1_server/emotions.py` mirrors
      `emotions.txt` in the HF repo. Re-sync if upstream changes.
- [ ] **Tune defaults** if needed — generation params in `config.py`
      (temperature/top_p/repetition_penalty) follow the model-card
      recommendations; adjust for your voice/domain.
- [ ] **Add authentication** — the server ships without auth. Put it behind a
      reverse proxy or add an API-key dependency before exposing it publicly.
- [ ] **(Optional) Streaming / scale** — implement the vLLM backend for
      continuous batching, prefix caching, and sub-100 ms streaming. Design
      outline in [docs/DEPLOYMENT.md §6](docs/DEPLOYMENT.md#6-concurrency--scaling).
      Keep the `load()`/`generate()` interface so the HTTP layer is untouched.
- [ ] **(Optional) Tests** — `audio.py` (SNAC unpack/encode) and `emotions.py`
      are pure and unit-testable without a GPU; `prompt.py` needs only a
      tokenizer. Add a `tests/` package if you want CI coverage.

### Design notes / deliberate scope

- **Single-process, serialised generation.** One GPU cannot run `generate`
  concurrently, so requests are locked. Scale with replicas or vLLM — see the
  deployment guide.
- **Reference backend = `transformers`.** Chosen for portability and a 1:1 match
  with the model card. vLLM is the documented production path, kept as a clean
  swap rather than a half-built feature.
- **No fabricated streaming.** The reference endpoint returns the full clip; true
  incremental streaming is a vLLM feature and is documented as such, not faked.

---

## License & attribution

- This server: use freely within your project.
- **Maya1 model**: Apache 2.0 — <https://huggingface.co/maya-research/maya1>
- **SNAC codec**: <https://huggingface.co/hubertsiuzdak/snac_24khz>
- Maya1 is developed at **Maya Research**. Project site: <https://maya1.org/>
