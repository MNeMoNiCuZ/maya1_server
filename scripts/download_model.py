#!/usr/bin/env python3
"""Pre-download the Maya1 model and SNAC codec into the HuggingFace cache.

Useful for baking weights into an image layer or warming a cache volume before
first request, so startup does not block on a ~6 GB download.

Usage (inside the container or a configured venv)::

    python scripts/download_model.py

Respects MAYA1_MODEL_REPO / MAYA1_SNAC_REPO and HF_TOKEN from the environment.
"""

from __future__ import annotations

import os
import sys

# Make the src/ layout importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maya1_server.constants import DEFAULT_MODEL_REPO, DEFAULT_SNAC_REPO  # noqa: E402


def main() -> None:
    model_repo = os.environ.get("MAYA1_MODEL_REPO", DEFAULT_MODEL_REPO)
    snac_repo = os.environ.get("MAYA1_SNAC_REPO", DEFAULT_SNAC_REPO)
    token = os.environ.get("HF_TOKEN")

    from huggingface_hub import snapshot_download

    print(f"[download] Fetching model weights: {model_repo}")
    snapshot_download(repo_id=model_repo, token=token)

    print(f"[download] Fetching SNAC codec: {snac_repo}")
    snapshot_download(repo_id=snac_repo, token=token)

    print("[download] Done. Weights are cached under HF_HOME / ~/.cache/huggingface.")


if __name__ == "__main__":
    main()
