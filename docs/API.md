# Maya1 Server — HTTP API Reference

Base URL (default): `http://localhost:41217`
Interactive docs (OpenAPI/Swagger): `http://localhost:41217/docs`
Machine-readable schema: `http://localhost:41217/openapi.json`

All request/response bodies are JSON except the synthesis endpoint, which
returns **binary audio**.

---

## `GET /`

Service metadata.

```json
{
  "name": "maya1-server",
  "version": "1.0.0",
  "backend": "gguf",
  "model_repo": "maya-research/maya1",
  "gguf_repo": "Mungert/maya1-GGUF",
  "gguf_filename": "maya1-q4_k_m.gguf",
  "snac_repo": "hubertsiuzdak/snac_24khz",
  "sample_rate": 24000,
  "description_format": "<description=\"...voice design...\"> text with <emotion> tags",
  "docs_url": "/docs"
}
```

`backend` is `"gguf"` (default, llama.cpp) or `"transformers"` depending on
`MAYA1_USE_GGUF`; `gguf_repo`/`gguf_filename` are only populated for the GGUF
backend. See [docs/DEPLOYMENT.md §4a](DEPLOYMENT.md#4a-backends-gguf-llamacpp-vs-transformers).

---

## `GET /health`

Liveness/readiness probe. Reports `loading` until the model finishes loading,
then `ok`. The Docker healthcheck waits for `"model_loaded": true`.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_repo": "maya-research/maya1",
  "backend": "gguf",
  "device": "cuda",
  "version": "1.0.0"
}
```

---

## `GET /v1/emotions`

Lists the supported inline emotion tags (served from the model's authoritative
list).

```json
{
  "emotions": ["laugh", "laugh_harder", "sigh", "..."],
  "tags": ["<laugh>", "<laugh_harder>", "<sigh>", "..."],
  "count": 17
}
```

---

## `POST /v1/audio/speech`

Synthesise speech. Returns raw audio bytes (not JSON). The request shape is
compatible with the OpenAI speech endpoint, plus a Maya1 `description` field.

### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `input` | string | **yes** | — | Text to speak. May contain inline emotion tags such as `<laugh>`. |
| `description` | string | no | server default | Natural-language voice design. See [MODEL.md](MODEL.md#2-voice-design-the-description). |
| `voice` | string | no | — | Alias for `description` (OpenAI compatibility). Used if `description` is absent. |
| `model` | string | no | — | Ignored; accepted for OpenAI client compatibility. |
| `response_format` | `"wav"` \| `"pcm"` | no | `"wav"` | `pcm` = raw 16-bit LE mono @ 24 kHz. |
| `max_new_tokens` | int (28–8192) | no | *auto-estimated* | Upper bound on generated tokens. Omit this to let the server estimate a safe budget from `input`'s length (see [Automatic length / `max_new_tokens`](#automatic-length-max_new_tokens) below) — this is what prevents long inputs from being cut off mid-word. Set explicitly only to override the estimate. |
| `min_new_tokens` | int (1–8192) | no | `28` | Lower bound (≥ 4 SNAC frames). |
| `temperature` | float (0–2) | no | `0.4` | Sampling temperature. |
| `top_p` | float (0–1) | no | `0.9` | Nucleus sampling. |
| `repetition_penalty` | float (1–2) | no | `1.1` | Anti-looping penalty. |

### Response

- **200** — binary body.
  - `Content-Type: audio/wav` (or `audio/L16; rate=24000; channels=1` for PCM).
  - Headers: `X-Audio-Duration-Seconds`, `X-Generated-Tokens`, `X-Elapsed-Seconds`.
- **422** — invalid input (empty, too long, or too few audio tokens produced).
- **503** — model still loading; retry shortly.
- **500** — unexpected backend error.

### Example — cURL

```bash
curl -X POST http://localhost:41217/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Female, in her 30s, American accent, energetic event host, clear diction.",
    "input": "Wow, this place looks amazing! <excited> I can not stop smiling right now.",
    "response_format": "wav"
  }' \
  --output hello.wav
```

### Example — Python (stdlib)

```python
import json, urllib.request

body = json.dumps({
    "description": "Male, late 20s, neutral American, warm baritone, calm pacing.",
    "input": "Welcome back to the show <chuckle> let's dive in.",
}).encode()

req = urllib.request.Request(
    "http://localhost:41217/v1/audio/speech",
    data=body, headers={"Content-Type": "application/json"}, method="POST",
)
with urllib.request.urlopen(req) as r:
    open("out.wav", "wb").write(r.read())
```

### Example — OpenAI Python SDK (compatibility mode)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:41217/v1", api_key="not-needed")

# `voice` carries the Maya1 description in compatibility mode.
resp = client.audio.speech.create(
    model="maya1",
    voice="Dark villain, male, 40s, British accent, low gravelly timbre, slow pacing.",
    input="You should not have come here <angry>.",
)
resp.stream_to_file("villain.wav")
```

> The `openai` SDK is a client-side convenience only; the server does not depend
> on it. Authentication is not enforced — put the server behind your own auth
> proxy if it is exposed beyond localhost.

---

## Automatic length / `max_new_tokens`

`max_new_tokens` is a hard cap on how many audio tokens a single generation
call is allowed to produce; the model stops early on its own once it emits an
end-of-speech token, but if it never does, generation runs until this cap and
the resulting audio is truncated mid-word.

When a request omits `max_new_tokens`, the server no longer just falls back to
a flat default — it estimates a safe budget from `input`'s character count
(`maya1_server/budget.py`), using an assumed speaking rate plus a safety
multiplier, then clamps the result to `MAYA1_MAX_NEW_TOKENS` (the
server-wide ceiling). Longer input text automatically gets a proportionally
larger budget, so you don't need to hand-tune this per request. Set
`max_new_tokens` explicitly only if you know better than the estimate (e.g. you
want a hard shorter clip, or you're intentionally pushing past the server
ceiling on a single call).

Tunable via environment variables (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `MAYA1_MAX_NEW_TOKENS` | `2048` | Ceiling the auto-estimate (and any explicit override) is clamped to. |
| `MAYA1_AUTO_TOKENS_CHARS_PER_SECOND` | `12.0` | Assumed speaking rate used for the estimate. Lower = more generous (larger) budgets. |
| `MAYA1_AUTO_TOKENS_HEADROOM` | `1.6` | Safety multiplier applied on top of the raw estimate. |
| `MAYA1_AUTO_TOKENS_FLOOR` | `256` | Minimum budget, even for very short input. |

What this does **not** solve: asking for many distinct lines/emotions inside
one long `input` string. That's a quality problem (drift, garbled audio as the
model loses coherence over a long multi-emotion passage), not a length-budget
problem — raising `max_new_tokens` further doesn't fix it. Use
`POST /v1/audio/speech/batch` for that instead (below).

---

## `POST /v1/audio/speech/batch`

Synthesise **multiple segments**, each as its own independent generation call
(own auto-estimated `max_new_tokens` unless overridden), then stitch the
results into a single audio file with a configurable silence gap between
segments. This is the server-side, documented replacement for "concatenate a
bunch of separate TTS calls yourself" — no client-side WAV parsing required.

Why per-segment calls instead of one big `input` string: Maya1's quality
degrades noticeably over long, multi-emotion passages generated in a single
pass (drift, garbling) — well before you'd hit any token limit. Short,
independent segments stitched together sound far better than one long
generation asked to cover the same ground.

### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `segments` | array (1–64) | **yes** | — | Ordered list of `{input, description?, ...per-segment overrides}` to synthesise and stitch. |
| `description` | string | no | server default | Voice design applied to any segment that doesn't set its own. |
| `voice` | string | no | — | Alias for `description`. |
| `gap_ms` | float (0–5000) | no | `300` | Silence inserted between segments, in milliseconds. |
| `response_format` | `"wav"` \| `"pcm"` | no | `"wav"` | Same as the single-segment endpoint. |
| `max_new_tokens`, `min_new_tokens`, `temperature`, `top_p`, `repetition_penalty` | — | no | — | Batch-level defaults for any segment that doesn't override them. Same semantics as the single-segment endpoint (`max_new_tokens` omitted = auto-estimated per segment). |

Each entry in `segments` accepts its own `input` (required), `description`
(optional, overrides the batch-level one), and its own
`max_new_tokens`/`min_new_tokens`/`temperature`/`top_p`/`repetition_penalty`
overrides — falling back to the batch-level value, then the server default.

### Response

- **200** — binary body, same content types as the single-segment endpoint.
  - `X-Audio-Duration-Seconds`, `X-Elapsed-Seconds` — for the combined clip.
  - `X-Segment-Count` — number of segments synthesised.
  - `X-Segment-Durations-Seconds` — comma-separated per-segment durations, in order.
  - `X-Segment-Tokens` — comma-separated per-segment generated-token counts, in order.
- **422** — a segment's `input` was empty/too long, or synthesis failed for a specific segment (error message names the index, e.g. `segments[3]: ...`).
- **503** / **500** — as above.

### Example — cURL

```bash
curl -X POST http://localhost:41217/v1/audio/speech/batch \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Female, in her 30s, American accent, warm and energetic, clear diction.",
    "gap_ms": 350,
    "segments": [
      {"input": "Hey there! This is Maya1 running in Docker and it sounds amazing."},
      {"input": "Laughing <laugh> that is honestly the funniest thing I have heard all week!"},
      {"input": "Angry <angry> I have told you three times already, this is unacceptable!"}
    ]
  }' \
  --output demo.wav
```
