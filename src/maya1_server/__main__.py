"""``python -m maya1_server`` entrypoint.

Starts a uvicorn server hosting the application from :mod:`maya1_server.app`.
Host/port and log level are taken from configuration (``MAYA1_*`` env vars).
"""

from __future__ import annotations

import uvicorn

from .config import settings
from .logging_setup import configure_logging, get_logger


def main() -> None:
    configure_logging(settings.log_level)
    logger = get_logger("maya1")
    logger.info("Launching uvicorn on %s:%d", settings.host, settings.port)
    uvicorn.run(
        "maya1_server.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # Single worker: the model lives in-process and is not fork-safe.
        workers=1,
    )


if __name__ == "__main__":
    main()
