#!/usr/bin/env python3
"""Minimal Python client for the Maya1 server.

Demonstrates a synthesis request and saves the resulting WAV. Uses only the
standard library so it runs anywhere without extra dependencies.

Usage::

    python scripts/example_client.py "Hello there <laugh> welcome back!" out.wav
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE_URL = os.environ.get("MAYA1_BASE_URL", "http://localhost:41217")

DEFAULT_DESCRIPTION = (
    "Male, late 20s, neutral American accent, warm baritone, calm pacing."
)


def synthesize(text: str, out_path: str, description: str = DEFAULT_DESCRIPTION) -> None:
    payload = json.dumps(
        {
            "description": description,
            "input": text,
            "response_format": "wav",
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{BASE_URL}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        duration = response.headers.get("X-Audio-Duration-Seconds", "?")
        audio_bytes = response.read()

    with open(out_path, "wb") as handle:
        handle.write(audio_bytes)

    print(f"Wrote {len(audio_bytes)} bytes ({duration}s of audio) to {out_path}")


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello! This is Maya1 <laugh> nice to meet you."
    out_path = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
    synthesize(text, out_path)


if __name__ == "__main__":
    main()
