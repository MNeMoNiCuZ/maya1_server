# Maya1 Server — Deployment Guide

How to run the server in Docker (recommended), configure it, and scale it.

---

## 1. Prerequisites

| Requirement | Why |
|---|---|
| NVIDIA GPU | **8 GB+ VRAM** with the default GGUF backend (`q4_k_m` quantization); **16 GB+ VRAM** if you switch to the full-precision `transformers` backend (`MAYA1_USE_GGUF=false`, see §4a). |
| Recent NVIDIA driver | Must support CUDA 12.8 (the image runtime). |
| Docker Engine 24+ / Docker Desktop | Container runtime. |
| **NVIDIA Container Toolkit** | Exposes the GPU to containers (`--gpus`). |
| ~10 GB free disk | Image + ~6 GB of downloaded weights. |

Verify GPU access from Docker before anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If that prints your GPU, you're ready. If not, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## 2. Quick start (Docker Compose)

```bash
cp .env.example .env         # optional: edit settings
docker compose up --build    # build image, download weights, start server
```

Watch the logs until you see `Startup complete; ready to serve`:

```bash
docker compose logs -f
```

Then test it:

```bash
scripts/smoke_test.sh        # hits /health, /v1/emotions, writes out.wav
```

Stop / restart:

```bash
docker compose down          # stop (weights persist in the hf-cache volume)
docker compose up            # start again (fast — weights already cached)
```

> **First start is slow**: it downloads ~6 GB of weights into the named volume
> `hf-cache`. The healthcheck `start_period` is 10 minutes to allow for this.
> Subsequent starts reuse the volume and are fast.

---

## 3. Quick start (plain Docker)

```bash
docker build -t maya1-server .

docker run --rm --gpus all \
  -p 41217:41217 \
  -v maya1-hf-cache:/root/.cache/huggingface \
  --env-file .env \
  maya1-server
```

Pre-download weights into the cache volume (optional, avoids first-request delay):

```bash
docker run --rm --gpus all \
  -v maya1-hf-cache:/root/.cache/huggingface \
  maya1-server python /app/scripts/download_model.py
```

---

## 4. Configuration

All settings are environment variables with the `MAYA1_` prefix (see
[`.env.example`](../.env.example) and
[`src/maya1_server/config.py`](../src/maya1_server/config.py)). The most useful:

| Variable | Default | Purpose |
|---|---|---|
| `MAYA1_PORT` | `41217` | HTTP port. |
| `MAYA1_USE_GGUF` | `true` | `true` = quantized GGUF via llama.cpp (default, fast, low VRAM). `false` = original full-precision `transformers` backend. See §4a. |
| `MAYA1_DEVICE` | `auto` | `auto` / `cuda` / `cpu`. |
| `MAYA1_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` (transformers backend only). |
| `MAYA1_WARMUP` | `true` | Warm the model at startup. |
| `MAYA1_TEMPERATURE` | `0.4` | Default sampling temperature. |
| `MAYA1_MAX_INPUT_CHARS` | `2000` | Reject overly long inputs. |
| `HF_TOKEN` | — | Only needed for gated/private model repos. |

Change a setting by editing `.env` and running `docker compose up -d` again.
`GET /health` and `GET /` both report which backend is active (`"backend":
"gguf"` or `"transformers"`).

---

## 4a. Backends: GGUF (llama.cpp) vs. transformers

Two interchangeable backends implement the exact same `load()`/`generate()`
interface (`src/maya1_server/gguf_engine.py` and `src/maya1_server/engine.py`);
`app.py` picks one at startup based on `MAYA1_USE_GGUF`. The HTTP layer,
`docker-compose.yml`, and every endpoint behave identically either way - only
startup time, VRAM use, and (very slightly) audio quality differ.

| | GGUF (default) | transformers |
|---|---|---|
| `MAYA1_USE_GGUF` | `true` | `false` |
| Weights | `Mungert/maya1-GGUF` (`MAYA1_GGUF_REPO`), quantization selected by `MAYA1_GGUF_QUANT` | `maya-research/maya1` full precision |
| Runtime | `llama-cpp-python` (compiled with CUDA in the Dockerfile) | `transformers` + `torch` |
| Speed / VRAM | Faster, far less VRAM (quantized) | Slower, needs the full 16 GB+ card |
| Quality | Small quantization-related quality cost (tunable - see quant table below) | Reference quality |

GGUF-specific settings:

| Variable | Default | Purpose |
|---|---|---|
| `MAYA1_GGUF_REPO` | `Mungert/maya1-GGUF` | HF repo hosting the GGUF files. |
| `MAYA1_GGUF_QUANT` | `q4_k_m` | Which quantization to download/load - just the short name; see the table below for all valid values. Invalid values fail fast at startup with the valid list in the error. |
| `MAYA1_GGUF_FILENAME_OVERRIDE` | unset | Advanced: an exact filename in `MAYA1_GGUF_REPO`, bypassing `MAYA1_GGUF_QUANT` entirely (e.g. a custom/third-party GGUF build). |
| `MAYA1_GGUF_N_CTX` | `8192` | llama.cpp context window; must fit prompt + `max_new_tokens`. |
| `MAYA1_GGUF_N_GPU_LAYERS` | `-1` | `-1` = offload all layers to GPU; `0` = CPU-only GGUF inference. |
| `MAYA1_GGUF_N_THREADS` | unset (auto) | CPU thread count for llama.cpp; only matters when not fully GPU-offloaded. |

### Available quantizations (`MAYA1_GGUF_QUANT`)

Full catalog lives in [`src/maya1_server/gguf_quants.py`](../src/maya1_server/gguf_quants.py)
(the single source of truth - the server validates `MAYA1_GGUF_QUANT` against
it at startup). Files are hosted at
[`Mungert/maya1-GGUF`](https://huggingface.co/Mungert/maya1-GGUF/tree/main);
smaller/earlier in this table = faster + less VRAM + lower quality, larger/later
= closer to full precision.

| `MAYA1_GGUF_QUANT` | File | Size | Notes |
|---|---|---|---|
| `q2_k_s` | [`maya1-q2_k_s.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q2_k_s.gguf) | 1.30 GB | Smallest, lowest quality. |
| `q2_k_m` | [`maya1-q2_k_m.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q2_k_m.gguf) | 1.36 GB | Very small, low quality. |
| `q3_k_s` | [`maya1-q3_k_s.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q3_k_s.gguf) | 1.69 GB | Small, noticeable quality loss. |
| `q3_k_m` | [`maya1-q3_k_m.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q3_k_m.gguf) | 1.75 GB | Small, moderate quality loss. |
| `q4_0` | [`maya1-q4_0.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q4_0.gguf) | 2.11 GB | Legacy 4-bit, superseded by `q4_k_*`. |
| `q4_1` | [`maya1-q4_1.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q4_1.gguf) | 2.10 GB | Legacy 4-bit, superseded by `q4_k_*`. |
| `q4_k_s` | [`maya1-q4_k_s.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q4_k_s.gguf) | 2.02 GB | Slightly smaller/faster than `q4_k_m`, a bit lower quality. |
| **`q4_k_m`** (default) | [`maya1-q4_k_m.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q4_k_m.gguf) | 2.19 GB | Good speed/quality/VRAM balance. |
| `q5_k_m` | [`maya1-q5_k_m.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q5_k_m.gguf) | 2.52 GB | Higher quality, more VRAM. |
| `q6_k_m` | [`maya1-q6_k_m.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q6_k_m.gguf) | 2.83 GB | Close to full precision, more VRAM. |
| `q8_0` | [`maya1-q8_0.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-q8_0.gguf) | 3.52 GB | Near-lossless, largest common quant. |
| `f16_q8_0` | [`maya1-f16_q8_0.gguf`](https://huggingface.co/Mungert/maya1-GGUF/blob/main/maya1-f16_q8_0.gguf) | 4.96 GB | Highest quality available, largest/slowest. |

To switch, e.g. to `q4_k_m`: set `MAYA1_GGUF_QUANT=q4_k_m` in `.env`, then
`docker compose up -d` (it will download that file into the `hf-cache` volume
on first use; other quants you've already downloaded stay cached alongside it).

**Why llama.cpp specifically, not a generic GGUF reader:** Maya1's generated
tokens past the header are raw SNAC audio-code IDs, not text — feeding them
through a naive text-based GGUF pathway (tokenize prompt → sample → detokenize
to a string → re-tokenize) corrupts that ID stream and produces garbled
"alien" noise instead of speech. `gguf_engine.py` avoids this by encoding the
prompt to raw token IDs once (same HF tokenizer `prompt.py` already uses),
feeding those IDs straight into `Llama.generate()`'s integer-token sampling
loop, and passing the raw sampled IDs directly into the same SNAC decode path
used by the transformers backend — never touching text in between. See that
file's module docstring for the full explanation, including why a
`transformers`-style logits processor (used in the transformers backend to
constrain sampling to valid SNAC tokens, see `logits.py`) must **not** be
ported to the GGUF backend — llama.cpp's sampling pipeline doesn't compose
with that kind of external logit masking; it breaks generation outright.

**Building for your GPU:** the Dockerfile compiles `llama-cpp-python` from
source (needs `nvcc`, hence the `-devel` base image) targeting
`CMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89;90;120` — Pascal through Blackwell.
This is what makes the GGUF backend work on older cards, not just RTX 50-series;
it does not *require* a 50-series GPU. If you only have one architecture, trim
this list in the Dockerfile to speed up the (one-time) image build.

---

## 5. GPU selection

Pin to a specific GPU by index with Compose:

```yaml
# docker-compose.yml → services.maya1.deploy.resources.reservations.devices
- driver: nvidia
  device_ids: ["0"]     # replace `count: all` with this to pin GPU 0
  capabilities: [gpu]
```

Or with plain Docker: `--gpus '"device=0"'`.

---

## 6. Concurrency & scaling

The reference `transformers` backend holds **one model in one process** and
serialises requests behind a lock (a single GPU cannot safely run `generate`
concurrently). Options to scale throughput:

1. **Horizontal replicas** — run N containers (one per GPU) behind a reverse
   proxy / load balancer (nginx, Traefik, a k8s Service). Each is stateless.
2. **Scaling with vLLM** *(recommended for production/streaming)* — swap the
   backend for vLLM, which supports continuous batching, automatic prefix
   caching (great when many requests reuse the same voice `description`), and
   sub-100 ms streaming. The model card ships a reference
   `vllm_streaming_inference.py`. Implementation outline:
   - Add a `Maya1VllmEngine` alongside `Maya1Engine` implementing the same
     `load()` / `generate()` interface.
   - Back it with an `AsyncLLMEngine`; stream token IDs out and run the same
     `unpack_snac_from_7` → SNAC decode incrementally per frame group.
   - Select the backend via a new `MAYA1_BACKEND=transformers|vllm` setting.
   - The HTTP layer (`api/routes.py`) and audio utilities need no changes.

> True low-latency **streaming** (chunked audio while generating) is a vLLM-path
> feature and is intentionally out of scope for the reference backend, which
> generates the full clip then returns it. This is called out honestly rather
> than half-implemented.

---

## 7. Running without Docker (local dev)

Requires a Python 3.10+ environment with a CUDA-matched `torch` installed
first (see <https://pytorch.org/get-started/locally/>), then:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m maya1_server           # reads MAYA1_* / .env
```

On Windows the repo's `venv_create.bat` / `launch_app_venv.bat` helpers set up a
local virtual environment; point the launcher at `python -m maya1_server`.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `no CUDA device is available` at startup | GPU not exposed to the container. Re-check the NVIDIA Container Toolkit and `--gpus all`. |
| Container stuck "unhealthy" for minutes | Normal on first run while ~6 GB downloads. Watch `docker compose logs -f`. |
| `503 Model still loading` | Weights not loaded yet; retry after `/health` reports `ok`. |
| OOM / CUDA out of memory | Use `MAYA1_DTYPE=float16`, close other GPU processes, or use a ≥16 GB GPU. |
| `422 too few audio tokens` | Input too short/odd; increase `max_new_tokens` or lengthen the text. |
| Flat/robotic audio | Raise `MAYA1_TEMPERATURE` toward 0.6–0.8. |
| Warbly/unstable audio | Lower temperature; keep `repetition_penalty` ≥ 1.1. |
| Garbled/"alien" noise, GGUF backend | Something is round-tripping generated tokens through text instead of keeping them as raw IDs end-to-end - see `gguf_engine.py`'s module docstring. Don't add a logits processor to the GGUF path (§4a); that also breaks it. |
| `llama-cpp-python` build fails / very slow image build | It's compiled from source against several CUDA architectures (§4a) - normal, one-time cost. Trim `CMAKE_CUDA_ARCHITECTURES` in the Dockerfile to just your GPU's arch to speed it up. |
