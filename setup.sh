#!/usr/bin/env bash
# Bare-metal (non-Docker) setup for the Maya1 TTS server on Linux.
#
# Creates a virtual environment (if one doesn't exist yet), activates it,
# and runs setup.py to install torch, requirements.txt, and llama-cpp-python
# (CUDA build) with guidance along the way.
#
#   ./setup.sh
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_NAME="${VENV_NAME:-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON_BIN' not found. Install Python 3 first, e.g.:"
    echo "  sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
    exit 1
fi

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment in ./$VENV_NAME ..."
    if ! "$PYTHON_BIN" -m venv "$VENV_NAME"; then
        echo
        echo "ERROR: venv creation failed. On Debian/Ubuntu this usually means"
        echo "the venv module isn't installed - fix with:"
        echo "  sudo apt-get update && sudo apt-get install -y python3-venv"
        exit 1
    fi
    echo "*" > "$VENV_NAME/.gitignore"
    echo "!.gitignore" >> "$VENV_NAME/.gitignore"
else
    echo "Reusing existing virtual environment ./$VENV_NAME"
fi

# shellcheck disable=SC1091
source "$VENV_NAME/bin/activate"

python -m pip install --upgrade pip
python setup.py

echo
echo "To reactivate this environment later:  source $VENV_NAME/bin/activate"
