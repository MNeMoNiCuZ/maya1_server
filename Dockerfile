# syntax=docker/dockerfile:1
#
# Maya1 TTS server image.
#
# Base: NVIDIA CUDA 12.8 **devel** on Ubuntu 22.04 (needs `nvcc`, not just the
# runtime libs, to compile llama-cpp-python's CUDA backend from source below -
# see docs/DEPLOYMENT.md). The host must provide the NVIDIA driver + the
# NVIDIA Container Toolkit.
#
# CUDA 12.8 (+ torch cu128 wheels, + llama-cpp-python built for a broad arch
# list below) is required for Blackwell GPUs (RTX 50-series, sm_120) - CUDA
# 12.1 only has kernels up to sm_90. Older (Pascal+) cards are still fully
# supported, not just Blackwell - see CMAKE_CUDA_ARCHITECTURES below.
#
# Build:  docker build -t maya1-server .
# Run:    docker run --gpus all -p 41217:41217 -v maya1-hf-cache:/root/.cache/huggingface maya1-server
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Model weights cache (mount a volume here to avoid re-downloading).
    HF_HOME=/root/.cache/huggingface \
    # Make the src/ package importable.
    PYTHONPATH=/app/src

# --- System Python + build essentials (cmake/g++ needed to build llama-cpp-python) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        cmake \
        build-essential \
        git \
        curl \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install torch first, from the CUDA 12.8 wheel index (Blackwell/sm_120 support) ---
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7"

# --- Install llama-cpp-python with CUDA support, built from source ---
# No prebuilt CUDA wheel is guaranteed to exist yet for every CUDA/arch
# combination (particularly Blackwell/sm_120), so this always builds from
# source against the devel image's own nvcc. CMAKE_CUDA_ARCHITECTURES lists
# Pascal (61) through Blackwell (120) explicitly - this is what makes the GGUF
# backend work on older cards too, not just 50-series. This step is slow
# (compiles for every listed arch) but is a one-time image-build cost.
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89;90;120"
RUN python -m pip install --no-cache-dir --force-reinstall --upgrade --verbose llama-cpp-python

# --- Install the rest of the Python dependencies ---
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# --- Application code ---
COPY src/ /app/src/
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh

EXPOSE 41217

# Container is healthy once the model has finished loading.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS "http://localhost:${MAYA1_PORT:-41217}/health" \
        | grep -q '"model_loaded":true' || exit 1

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
