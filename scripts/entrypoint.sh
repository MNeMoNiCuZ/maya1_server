#!/usr/bin/env bash
#
# Container entrypoint. Launches the Maya1 server via the package __main__,
# which reads MAYA1_* configuration from the environment.
#
# Any arguments passed to `docker run ... <args>` are forwarded verbatim, so you
# can override the command, e.g.:
#   docker run --gpus all maya1-server python /app/scripts/download_model.py
set -euo pipefail

if [[ $# -gt 0 ]]; then
    exec "$@"
fi

echo "[entrypoint] Starting Maya1 server (device=${MAYA1_DEVICE:-auto}, dtype=${MAYA1_DTYPE:-bfloat16})"
exec python -m maya1_server
