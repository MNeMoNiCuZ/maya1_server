"""Maya1 smoke test - interactive, pure Python (no PowerShell).

Checks /health and /v1/emotions, then synthesises a WAV file. You are
prompted for a voice description and spoken text; leave either blank to
fall back to the defaults configured below. Output is never overwritten:
output1.wav, output2.wav, output3.wav, ... (first free name wins).

If you leave the spoken-text prompt blank, the default demo runs the intro
line, then EVERY emotion below, via a single call to the server's
POST /v1/audio/speech/batch endpoint - the server generates each line as
its own short, independent request (this matters: asking the model for one
long multi-emotion clip in a single generation degrades badly) and stitches
them into one WAV with a silence gap between each. See docs/API.md for the
full endpoint reference.

Run directly:  python scripts\\smoke_test.py
Or just double-click sample_generation.bat, which launches this.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ============================================================================
# CONFIG - tweak these, nothing below this section needs to change.
# ============================================================================

# Which GGUF quantization you expect the server to be running (must match the
# server's own MAYA1_GGUF_QUANT in .env - this script does NOT change server
# config, it only checks and warns you on a mismatch). See
# src/maya1_server/gguf_quants.py / docs/DEPLOYMENT.md §4a for the full list:
#   q2_k_s  q2_k_m  q3_k_s  q3_k_m  q4_0  q4_1  q4_k_s  q4_k_m (default)
#   q5_k_m  q6_k_m  q8_0  f16_q8_0
EXPECTED_GGUF_QUANT = "q4_k_m"

# Used when you leave the voice-description prompt blank.
DEFAULT_VOICE_DESCRIPTION = (
    "Female, in her 30s with an American accent, warm and energetic, clear diction."
)

# Spoken first, before the emotion run-through, in the default demo.
INTRO_LINE = "Hey there! This is Maya1 running in Docker and it sounds amazing."

# One line per emotion tag: the words Maya1 will actually speak for that
# emotion. Edit, add, or remove entries here to change the demo. Each is sent
# to the server as its own batch segment (see note above), so keep these short.
EMOTION_SENTENCES = {
    "laugh": "Laughing <laugh> that is honestly the funniest thing I've heard all week!",
    "laugh_harder": "Laughing harder <laugh_harder> stop, stop, I can't breathe, this is too much!",
    "sigh": "Sighing <sigh> I suppose we'll just have to try again tomorrow.",
    "chuckle": "Chuckling <chuckle> well, that's one way to put it.",
    "gasp": "Gasping <gasp> oh my goodness, I did not expect that at all!",
    "angry": "Angry <angry> I have told you three times already, this is unacceptable!",
    "excited": "Excited <excited> we finally shipped the feature and it works perfectly!",
    "whisper": "Whispering <whisper> don't make a sound, someone is right outside the door.",
    "cry": "Crying <cry> I just really miss how things used to be.",
    "scream": "Screaming <scream> watch out, it's falling right toward you!",
    "sing": "Singing <sing> oh what a beautiful morning, oh what a beautiful day.",
    "snort": "Snorting <snort> ha, yeah right, like that's ever going to happen.",
    "exhale": "Exhaling <exhale> okay, we made it, that's finally over.",
    "gulp": "Gulping <gulp> um, I think we might be in a little bit of trouble here.",
    "giggle": "Giggling <giggle> hehe, sorry, I just find this whole thing ridiculous.",
    "sarcastic": "Sarcastic <sarcastic> oh sure, because that plan worked out so well last time.",
    "curious": "Curious <curious> hmm, I wonder what's actually inside that box.",
}

# Silence inserted between segments by the server for the default demo.
GAP_MILLISECONDS = 350

# Base filename for the synthesised clip (output1.wav, output2.wav, ...).
# Always written next to the repo root (one level up from scripts\), regardless
# of where you ran it from.
OUTPUT_BASE_NAME = "output"
OUTPUT_EXTENSION = ".wav"
OUTPUT_DIR = Path(__file__).resolve().parent.parent

BASE_URL = os.environ.get("MAYA1_BASE_URL", "http://localhost:41217")

# ============================================================================
# Script body
# ============================================================================


def get_next_output_file(directory: Path, base_name: str, extension: str) -> Path:
    i = 1
    while (directory / f"{base_name}{i}{extension}").exists():
        i += 1
    return directory / f"{base_name}{i}{extension}"


def http_get_json(path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def show_http_error(exc: Exception) -> None:
    print(f"Request failed: {exc}")
    if isinstance(exc, urllib.error.HTTPError):
        try:
            print(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            pass


class SynthResult:
    def __init__(self):
        self.headers: dict = {}
        self.error: Exception | None = None


def synth_with_progress(uri: str, body: dict, out_file: Path) -> dict:
    """POST `body` to `uri`, save the response body to `out_file`.

    The synth endpoints block for the whole generation with no progress of
    their own, so this runs the request on a background thread and prints an
    elapsed timer here - plus a periodic /health ping (served on its own
    thread even while a generation request is in flight) so you can tell
    "still working" from "server died" instead of staring at a silent
    console.
    """
    result = SynthResult()

    def worker():
        try:
            req = urllib.request.Request(
                uri,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=None) as resp:
                out_file.write_bytes(resp.read())
                result.headers = {k.lower(): v for k, v in resp.headers.items()}
        except Exception as exc:  # noqa: BLE001 - surfaced to the main thread below
            result.error = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    t0 = time.perf_counter()
    last_health_check = 0.0
    while thread.is_alive():
        time.sleep(2)
        elapsed = time.perf_counter() - t0

        health_note = ""
        if elapsed - last_health_check >= 10:
            last_health_check = elapsed
            try:
                h = http_get_json("/health", timeout=5)
                health_note = f" | server responsive (model_loaded={h.get('model_loaded')})"
            except Exception:
                health_note = " | WARNING: server did not respond to /health just now"

        print(f"  ... still waiting on server ({elapsed:.0f}s elapsed){health_note}")

    thread.join()
    print(f"  Server responded after {time.perf_counter() - t0:.0f}s.")

    if result.error is not None:
        raise result.error
    return result.headers


def main() -> int:
    print("== 1. Health ==")
    try:
        health = http_get_json("/health")
        print(json.dumps(health))
    except Exception as exc:
        show_http_error(exc)
        input("Press Enter to exit")
        return 1

    device = health.get("device", "unknown")
    backend = health.get("backend", "unknown")
    print(f"  Device: {device}  (backend={backend})")
    if device == "cpu":
        print(
            "  WARNING: server is running on CPU - generation will be extremely slow. "
            "See README.md 'Installing torch manually' if you expected GPU."
        )

    if backend == "gguf":
        try:
            info = http_get_json("/")
            gguf_filename = info.get("gguf_filename")
            print(f"  Model: {gguf_filename} (backend=gguf)")
            if gguf_filename and gguf_filename != f"maya1-{EXPECTED_GGUF_QUANT}.gguf":
                print(
                    f"  NOTE: server is running '{gguf_filename}', not the "
                    f"'{EXPECTED_GGUF_QUANT}' this script expected (EXPECTED_GGUF_QUANT "
                    "at the top) - not an error, just a heads-up."
                )
        except Exception:
            print("  Model: (backend=gguf, could not fetch filename from '/')")
    else:
        print("  Model: (backend=transformers, full-precision)")
    print()

    print("== 2. Supported emotions ==")
    try:
        emotions_resp = http_get_json("/v1/emotions")
        print(json.dumps(emotions_resp))
    except Exception as exc:
        print(f"Could not fetch emotions: {exc}")
    print()

    print("== 3. Voice + text ==")
    voice_input = input("Voice description (leave blank for default): ").strip()
    if not voice_input:
        voice_description = DEFAULT_VOICE_DESCRIPTION
        print("Using default voice description.")
    else:
        voice_description = voice_input

    text_input = input(
        "Spoken text (leave blank to run intro + all-emotions demo): "
    ).strip()
    print()

    out_file = get_next_output_file(OUTPUT_DIR, OUTPUT_BASE_NAME, OUTPUT_EXTENSION)

    try:
        if not text_input:
            segments = [{"input": INTRO_LINE}]
            segments += [{"input": sentence} for sentence in EMOTION_SENTENCES.values()]

            print(
                f"== 4. Synthesising intro + {len(EMOTION_SENTENCES)} emotion lines via "
                f"/v1/audio/speech/batch -> {out_file} =="
            )
            print(
                f"  (this runs {len(segments)} short generations server-side and stitches "
                "them; watch 'docker compose logs -f' for per-segment progress too)"
            )
            body = {
                "description": voice_description,
                "segments": segments,
                "gap_ms": GAP_MILLISECONDS,
                "response_format": "wav",
            }
            headers = synth_with_progress(
                f"{BASE_URL}/v1/audio/speech/batch", body, out_file
            )

            print(f"Saved combined audio to {out_file}")
            for h in ("x-segment-count", "x-audio-duration-seconds", "x-elapsed-seconds"):
                if h in headers:
                    print(f"{h}: {headers[h]}")
        else:
            print(f"== 4. Synthesise -> {out_file} ==")
            body = {
                "description": voice_description,
                "input": text_input,
                "response_format": "wav",
            }
            headers = synth_with_progress(f"{BASE_URL}/v1/audio/speech", body, out_file)

            print(f"Saved audio to {out_file}")
            for h in ("x-audio-duration-seconds", "x-generated-tokens", "x-elapsed-seconds"):
                if h in headers:
                    print(f"{h}: {headers[h]}")
    except Exception as exc:
        show_http_error(exc)
        input("Press Enter to exit")
        return 1

    print()
    print("Smoke test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
