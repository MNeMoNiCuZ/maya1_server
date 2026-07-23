# Maya1 — Model Reference

Everything an implementer or caller needs to know about the underlying model.
This is a distilled, server-oriented reference; the authoritative source is the
[`maya-research/maya1`](https://huggingface.co/maya-research/maya1) model card.

---

## 1. What Maya1 is

**Maya1** is an open-source, expressive **text-to-speech (TTS)** model from
**Maya Research**, released under the **Apache 2.0** license (commercial use
permitted). It is designed for two things existing open TTS models do poorly:

1. **Voice design from natural language** — you describe the voice you want in
   plain English (like briefing a voice actor) instead of picking from a fixed
   set of speaker IDs.
2. **Inline emotion control** — you place emotion tags such as `<laugh>` or
   `<whisper>` directly inside the text, exactly where the emotion should occur.

| Property | Value |
|---|---|
| Parameters | ~3 billion |
| Architecture | Decoder-only transformer, Llama-style |
| Output | Predicts **SNAC** neural-codec tokens (not raw waveform) |
| Audio codec | [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz) |
| Sample rate | 24 kHz, mono, 16-bit |
| Bitrate (stream) | ~0.98 kbps |
| Language | English |
| License | Apache 2.0 |
| VRAM | 16 GB+ full precision; **8 GB+ with the default 4-bit GGUF backend** (see below) |
| Dependencies | `torch`, `snac`, `soundfile`, plus either `transformers` (full precision) or `llama-cpp-python` (default, GGUF) |

This server ships **two interchangeable backends** (see
[docs/DEPLOYMENT.md §4a](DEPLOYMENT.md#4a-backends-gguf-llamacpp-vs-transformers)):
a quantized **GGUF** model run through `llama.cpp` (`MAYA1_USE_GGUF=true`,
the default — faster, far less VRAM) via
[`Mungert/maya1-GGUF`](https://huggingface.co/Mungert/maya1-GGUF)
(`maya1-q4_k_m.gguf` by default), and the original full-precision
**`transformers`** backend (`MAYA1_USE_GGUF=false`). Only the language model
backbone is quantized in the GGUF path — the SNAC decoder below is identical,
unquantized, in both.

The model predicts **7 SNAC tokens per audio frame**, which are unpacked into a
3-level hierarchical codebook and decoded to a 24 kHz waveform by the SNAC
decoder.

---

## 2. Voice design (the `description`)

The voice is controlled by a free-form natural-language **description**. There
is no fixed schema, but the model responds well to descriptions that mention:

- **Identity / character**: `event host`, `dark villain`, `mythical goddess`, `demon`
- **Gender & age**: `female, in her 30s`, `male, late 20s`
- **Accent**: `American`, `British`, `Middle Eastern`
- **Pitch**: `low pitch`, `normal pitch`, `high pitch`
- **Timbre**: `warm`, `gravelly`, `bright`, `baritone`
- **Pacing**: `slow pacing`, `conversational pacing`, `fast`
- **Tone & intensity**: `angry tone at high intensity`, `curious tone at medium intensity`

### Examples (from the model card)

```text
Female, in her 30s with an American accent and is an event host, energetic, clear diction
```
```text
Male, late 20s, neutral American, warm baritone, calm pacing
```
```text
Dark villain character, Male voice in their 40s with a British accent. low pitch, gravelly timbre, slow pacing, angry tone at high intensity.
```
```text
Demon character, Male voice in their 30s with a Middle Eastern accent. screaming tone at high intensity.
```
```text
Mythical godlike magical character, Female voice in their 30s slow pacing, curious tone at medium intensity.
```

> The description sets the *persona* for the whole utterance. Emotion **tags**
> (below) drive *moment-to-moment* expression within the text.

---

## 3. Emotion tags

Maya1 supports **17 inline emotion tags** (the model card rounds this to
"20+"). Place a tag inline in the text; it applies to the surrounding speech.

```text
Our new update <laugh> finally ships the feature you asked for.
```

### Full supported list

| Tag | Tag | Tag | Tag |
|---|---|---|---|
| `<laugh>` | `<laugh_harder>` | `<sigh>` | `<chuckle>` |
| `<gasp>` | `<angry>` | `<excited>` | `<whisper>` |
| `<cry>` | `<scream>` | `<sing>` | `<snort>` |
| `<exhale>` | `<gulp>` | `<giggle>` | `<sarcastic>` |
| `<curious>` | | | |

This exact list is served live by the running server at **`GET /v1/emotions`**
and is defined in [`src/maya1_server/emotions.py`](../src/maya1_server/emotions.py).
It mirrors `emotions.txt` in the HuggingFace repo (the authoritative source).

### Usage guidance

- Tags are written **lowercase inside angle brackets**, matching the list above.
- Place a tag **immediately before** the words it should color, or between
  sentences for a standalone expression.
- Unknown tags (e.g. a typo like `<laughs>`) are **not rejected** — the model
  will try to interpret them — but the server logs a warning when
  `MAYA1_WARN_UNKNOWN_TAGS=true` so typos surface early.
- Combine sparingly. One or two tags per sentence reads naturally; stacking many
  tends to destabilise the audio.

---

## 4. Prompt format (how the server frames a request)

The model was trained on a specific token framing. The server builds it for you
in [`src/maya1_server/prompt.py`](../src/maya1_server/prompt.py); you never write
it by hand. Conceptually:

```
<SOH> <BOS> <description="...voice..."> <text with <emotion> tags> <EOT> <EOH> <SOA> <SOS>
```

The human-readable core is:

```text
<description="Realistic male voice in the 30s ... warm timbre, conversational pacing."> Hello! This is Maya1 <laugh_harder> the best open source voice AI model with emotions.
```

The model then emits SNAC audio tokens until it produces the end-of-speech
token.

### Special token IDs

These are **fixed by the model** — they are not configurable and live in
[`src/maya1_server/constants.py`](../src/maya1_server/constants.py):

| Name | ID | Meaning |
|---|---|---|
| `BOS_ID` | 128000 | Beginning of sequence |
| `TEXT_EOT_ID` | 128009 | End of text turn |
| `CODE_START_TOKEN_ID` | 128257 | Start of speech (SOS) |
| `CODE_END_TOKEN_ID` | 128258 | End of speech (EOS) — generation stop token |
| `SOH_ID` | 128259 | Start of header |
| `EOH_ID` | 128260 | End of header |
| `SOA_ID` | 128261 | Start of audio |
| `CODE_TOKEN_OFFSET` | 128266 | SNAC codes are offset by this value |
| SNAC token range | 128266–156937 | Valid audio-code tokens |

---

## 5. Generation parameters

Model-card recommended defaults (also the server defaults, overridable per
request and via `MAYA1_*` env vars):

| Parameter | Default | Notes |
|---|---|---|
| `temperature` | `0.4` | Low → stable, consistent delivery. Raise for variety. |
| `top_p` | `0.9` | Nucleus sampling. |
| `repetition_penalty` | `1.1` | Prevents looping/stuttering audio. |
| `max_new_tokens` | `2048` | Upper bound; the model stops early at EOS. |
| `min_new_tokens` | `28` | ≥ 4 SNAC frames, so very short inputs still decode. |
| `do_sample` | `true` | Sampling is required for natural prosody. |
| `eos_token_id` | `128258` | Stop at end-of-speech. |

**Tuning tips**

- Robotic / flat delivery → raise `temperature` toward `0.6–0.8`.
- Unstable / warbly audio → lower `temperature`, keep `repetition_penalty` ≥ 1.1.
- Truncated endings → raise `max_new_tokens`.

---

## 6. SNAC decoding pipeline

The model outputs a flat token stream; converting it to audio (implemented in
[`src/maya1_server/audio.py`](../src/maya1_server/audio.py)):

1. **Extract** — keep tokens in the SNAC range `[128266, 156937]`, up to the
   end-of-speech token.
2. **Unpack 7 → 3 levels** — each 7-token frame maps to:
   - **L1**: 1 code (slot 0)
   - **L2**: 2 codes (slots 1, 4)
   - **L3**: 4 codes (slots 2, 3, 5, 6)

   Each code is de-offset (`- 128266`) and taken `% 4096`.
3. **Decode** — `snac.quantizer.from_codes(...)` → `snac.decoder(...)` yields a
   float32 waveform.
4. **Trim warmup** — drop the first **2048** samples (decoder ramp-up).
5. **Encode** — serialise to 24 kHz mono 16-bit WAV (or raw PCM).

---

## 7. Hardware & performance

- **Minimum**: single GPU with **16 GB VRAM** (RTX 4090). Runs comfortably on
  A100 / H100.
- **Precision**: `bfloat16` is recommended and default. Use `float16` on GPUs
  without bf16 support. `float32` roughly doubles VRAM.
- **CPU**: supported (`MAYA1_DEVICE=cpu`) for smoke tests only — it is far too
  slow for real use.
- **Latency**: the reference `transformers` backend generates the whole clip,
  then decodes. For sub-100 ms streaming latency, use the vLLM backend
  (see [`DEPLOYMENT.md`](DEPLOYMENT.md) → *Scaling with vLLM*).

---

## 8. Attribution & links

- Model: <https://huggingface.co/maya-research/maya1> (Apache 2.0)
- Codec: <https://huggingface.co/hubertsiuzdak/snac_24khz>
- Project site: <https://maya1.org/>
- Developed at Maya Research by Dheemanth Reddy Bhumireddy Singa Reddy and
  Bharath Kumar Kakumani.
