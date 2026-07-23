"""Maya1 TTS inference server.

A standards-based FastAPI server that wraps the ``maya-research/maya1`` speech
model for expressive, voice-designed text-to-speech synthesis.

The package is split by responsibility:

* :mod:`maya1_server.config`        - runtime configuration (env-driven).
* :mod:`maya1_server.constants`     - model special-token IDs and audio constants.
* :mod:`maya1_server.emotions`      - the emotion-tag registry.
* :mod:`maya1_server.prompt`        - prompt construction for the model.
* :mod:`maya1_server.audio`         - SNAC token unpacking and WAV/PCM encoding.
* :mod:`maya1_server.engine`        - model loading and generation.
* :mod:`maya1_server.app`           - FastAPI application factory.
* :mod:`maya1_server.api`           - request/response schemas and HTTP routes.
"""

__version__ = "1.0.0"
