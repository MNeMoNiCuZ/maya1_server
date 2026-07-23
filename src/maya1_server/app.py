"""FastAPI application factory and process entrypoint.

``create_app()`` wires configuration, logging, the engine and the routes
together. Model loading happens in the lifespan startup so the container's
health probe reports ``loading`` until weights are ready, then ``ok``.

Run with either::

    python -m maya1_server            # uses this module's __main__
    uvicorn maya1_server.app:app      # ASGI server picks up the module-level app
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api.routes import router
from .config import Settings, settings
from .logging_setup import configure_logging, get_logger


def create_app(app_settings: Settings | None = None) -> FastAPI:
    """Build and return the configured FastAPI application."""
    app_settings = app_settings or settings
    configure_logging(app_settings.log_level)
    logger = get_logger("maya1.app")

    if app_settings.use_gguf:
        from .gguf_engine import Maya1GGUFEngine
        logger.info("Backend: llama.cpp/GGUF (MAYA1_USE_GGUF=true)")
        engine = Maya1GGUFEngine(app_settings)
    else:
        from .engine import Maya1Engine
        logger.info("Backend: transformers (MAYA1_USE_GGUF=false)")
        engine = Maya1Engine(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Maya1 server v%s starting up", __version__)
        # Blocking load is intentional: the health probe reports 'loading' via
        # engine.is_loaded until this returns, then flips to 'ok'.
        engine.load()
        logger.info("Startup complete; ready to serve on %s:%d",
                    app_settings.host, app_settings.port)
        yield
        logger.info("Maya1 server shutting down")

    app = FastAPI(
        title="Maya1 TTS Server",
        version=__version__,
        summary="Expressive, voice-designed text-to-speech powered by maya-research/maya1.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = app_settings
    app.state.engine = engine
    app.include_router(router)
    return app


# Module-level ASGI app for `uvicorn maya1_server.app:app`.
app = create_app()
