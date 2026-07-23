"""Run the Maya1 TTS server directly with ``python server.py`` (no Docker).

Equivalent to ``python -m maya1_server``, provided at the repo root so the
project can be started without installing the package or setting
``PYTHONPATH`` manually. Requires the dependencies in requirements.txt (and a
CUDA-enabled torch/llama-cpp-python) to already be installed, e.g. via
venv_create.bat.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from maya1_server.__main__ import main

if __name__ == "__main__":
    main()
