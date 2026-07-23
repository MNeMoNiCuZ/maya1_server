"""Bare-metal (non-Docker) setup helper for the Maya1 TTS server.

Run this once inside an already-created and activated virtual environment
(see ``venv_create.bat`` on Windows or ``setup.sh`` on Linux):

    python setup.py

What it does, in order:

1. Copies ``.env.example`` to ``.env`` if ``.env`` doesn't exist yet.
2. Installs a CUDA-matched build of torch (cu128 wheel index), same as the
   Dockerfile, unless torch is already importable.
3. Installs the rest of requirements.txt.
4. Offers to build ``llama-cpp-python`` with CUDA support from source (this
   is the GGUF backend used by default, MAYA1_USE_GGUF=true) - or prints the
   manual steps + prerequisites for your OS if you'd rather do it yourself.

This mirrors what the Dockerfile does, just on your host instead of inside a
container. It is interactive and safe to re-run.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
# Pascal (61) through Blackwell (120) - the full range the Dockerfile builds
# for, so a container image works on any GPU. Bare-metal doesn't need that:
# we only ever need the *installed* GPU(s), which _target_cuda_architectures()
# detects, making the build both correct and dramatically faster (one arch
# instead of eight).
ALL_CUDA_ARCHITECTURES = [61, 70, 75, 80, 86, 89, 90, 120]


def _nvcc_version() -> tuple[int, int] | None:
    """Return (major, minor) of the installed CUDA toolkit, or None if nvcc isn't found."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    try:
        output = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", output)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


ARCH_LABELS = {
    61: "Pascal - e.g. GTX 10-series",
    70: "Volta - e.g. Titan V",
    75: "Turing - e.g. RTX 20-series / GTX 16-series",
    80: "Ampere (datacenter) - e.g. A100",
    86: "Ampere (consumer) - e.g. RTX 30-series",
    89: "Ada Lovelace - e.g. RTX 40-series",
    90: "Hopper - e.g. H100",
    120: "Blackwell - e.g. RTX 50-series",
}


def _detect_gpus() -> list[tuple[str, int]] | None:
    """Ask nvidia-smi which GPU(s) are actually installed.

    Returns e.g. [("NVIDIA GeForce RTX 5090", 120)]. None if nvidia-smi isn't
    available/parseable (no NVIDIA driver, or ran outside a machine with a
    GPU) - the caller falls back to asking the user or building broadly.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        output = subprocess.run(
            [nvidia_smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return None

    gpus: list[tuple[str, int]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.rsplit(",", 1)]
        if len(parts) != 2:
            continue
        name, cap = parts
        match = re.match(r"^(\d+)\.(\d+)$", cap)
        if match:
            gpus.append((name, int(match.group(1)) * 10 + int(match.group(2))))
    return gpus or None


def _supported_cuda_architectures() -> tuple[list[int], tuple[int, int] | None]:
    """Trim ALL_CUDA_ARCHITECTURES to what the installed nvcc actually supports.

    CUDA 13.0+ removed everything older than Turing (sm_75) - compiling
    compute_61/compute_70 with a CUDA 13 nvcc fails with
    "nvcc fatal: Unsupported gpu architecture". If nvcc can't be found/parsed,
    fall back to the full list and let the build itself surface any error.
    """
    version = _nvcc_version()
    if version is None:
        return list(ALL_CUDA_ARCHITECTURES), None
    major, _minor = version
    min_arch = 75 if major >= 13 else 0
    return [arch for arch in ALL_CUDA_ARCHITECTURES if arch >= min_arch], version


def _prompt_architecture_choice(nvcc_supported: list[int]) -> list[int]:
    """Let the user pick their GPU's architecture from a menu, when it
    couldn't be auto-detected or the user says the detected GPU is wrong.
    """
    print("\nWhich GPU architecture do you have?")
    for i, arch in enumerate(nvcc_supported, start=1):
        label = ARCH_LABELS.get(arch, f"sm_{arch}")
        print(f"  {i}. {label} (sm_{arch})")
    print(f"  0. Not sure / build for all of the above (slower, but works on any of these GPUs)")

    while True:
        reply = input(f"Select a number (Press Enter for default '0'): ").strip()
        if not reply:
            return nvcc_supported
        if reply.isdigit():
            choice = int(reply)
            if choice == 0:
                return nvcc_supported
            if 1 <= choice <= len(nvcc_supported):
                return [nvcc_supported[choice - 1]]
        print("Invalid choice, try again.")


def _target_cuda_architectures() -> tuple[list[int], tuple[int, int] | None, list[tuple[str, int]] | None]:
    """Pick the architecture list to actually build for, confirming with the
    user first.

    Prefers the real, installed GPU(s) (via nvidia-smi) intersected with what
    the toolkit supports - this is what makes the build fast (minutes, not
    hours) instead of compiling every kernel for all 8 architectures. The
    user confirms the detection (or corrects it) before it's used; if
    detection fails entirely, they pick from a menu or fall back to the full
    nvcc-supported list.
    """
    nvcc_supported, nvcc_ver = _supported_cuda_architectures()
    detected = _detect_gpus()

    if detected:
        names = ", ".join(f"{name} (sm_{cap})" for name, cap in detected)
        print(f"Detected GPU(s) via nvidia-smi: {names}")
        if _ask_yes_no("Is this correct?"):
            nvcc_min = nvcc_supported[0] if nvcc_supported else 0
            targeted = sorted({cap for _, cap in detected if cap >= nvcc_min})
            return (targeted or nvcc_supported), nvcc_ver, detected
        print("OK - pick your GPU's architecture manually instead.")
    else:
        print("Could not detect a GPU via nvidia-smi (not on PATH, or no "
              "NVIDIA driver found).")

    return _prompt_architecture_choice(nvcc_supported), nvcc_ver, detected


def _print_header(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    reply = input(f"{prompt} {suffix} ").strip().lower()
    if not reply:
        return default
    return reply in ("y", "yes")


def _run(cmd: list[str]) -> bool:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def step_env_file() -> None:
    _print_header("Step 1/4: .env file")
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"
    if env_path.exists():
        print(".env already exists - leaving it untouched.")
        return
    if not example_path.exists():
        print("WARNING: .env.example not found, cannot create .env. "
              "Create one manually before starting the server.")
        return
    shutil.copyfile(example_path, env_path)
    print(f"Created {env_path} from .env.example. Defaults work out of the "
          "box - edit it later to tune the port, model, quantization, etc.")


def step_torch() -> None:
    _print_header("Step 2/4: torch (CUDA build)")
    try:
        import torch  # noqa: F401

        print(f"torch is already installed (version {torch.__version__}) - skipping.")
        return
    except ImportError:
        pass

    print("torch is not installed. This project needs a CUDA 12.8-matched "
          "build (same as the Dockerfile) to use the GPU.")
    if not _ask_yes_no(f"Install torch from {TORCH_INDEX_URL} now?"):
        print("Skipped. Install it yourself before starting the server, e.g.:\n"
              f"  pip install --index-url {TORCH_INDEX_URL} \"torch>=2.7\"\n"
              "(pick a build matching your CUDA driver - see "
              "https://pytorch.org/get-started/locally/)")
        return
    ok = _run([sys.executable, "-m", "pip", "install",
               "--index-url", TORCH_INDEX_URL, "torch>=2.7"])
    if not ok:
        print("torch install failed - see the error above. You can retry "
              "manually with the command shown, or install a build matching "
              "your CUDA driver from https://pytorch.org/get-started/locally/")


def step_requirements() -> None:
    _print_header("Step 3/4: requirements.txt")
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        print("requirements.txt not found - skipping.")
        return
    if not _ask_yes_no("Install the rest of requirements.txt now?"):
        print("Skipped.")
        return
    ok = _run([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
    if not ok:
        print("requirements.txt install failed - see the error above.")


def _llama_cpp_prereqs_text() -> str:
    system = platform.system()
    if system == "Windows":
        return (
            "Prerequisites (Windows):\n"
            "  - NVIDIA CUDA Toolkit 12.8 (provides nvcc): "
            "https://developer.nvidia.com/cuda-downloads\n"
            "  - Visual Studio 2022 Build Tools with the \"Desktop development "
            "with C++\" workload: https://visualstudio.microsoft.com/downloads/\n"
            "  - CMake (if not already on PATH): https://cmake.org/download/\n"
            "  - Run this from a \"Developer Command Prompt for VS 2022\" (or "
            "a regular prompt with vcvarsall.bat/cl.exe on PATH), otherwise "
            "the build cannot find the MSVC compiler."
        )
    if system == "Linux":
        return (
            "Prerequisites (Linux):\n"
            "  - NVIDIA CUDA Toolkit 12.8 (provides nvcc): "
            "https://developer.nvidia.com/cuda-downloads\n"
            "  - build-essential + cmake:\n"
            "      sudo apt-get update && sudo apt-get install -y build-essential cmake"
        )
    return (
        "Prerequisites: an NVIDIA CUDA toolkit (nvcc) and a C++ build chain "
        "(cmake + a C++ compiler) matching your platform. This project "
        "targets Linux/Windows + CUDA; macOS/CPU-only is untested."
    )


def step_llama_cpp_python() -> None:
    _print_header("Step 4/4: llama-cpp-python (CUDA / GGUF backend)")
    try:
        import llama_cpp  # noqa: F401

        print("llama-cpp-python is already installed - skipping.")
        return
    except ImportError:
        pass

    print("llama-cpp-python powers the default GGUF backend "
          "(MAYA1_USE_GGUF=true) and needs to be built from source with CUDA "
          "support - there's no universal prebuilt CUDA wheel for every "
          "CUDA/GPU-architecture combination.\n")
    print(_llama_cpp_prereqs_text())
    print()

    architectures, nvcc_ver, detected_gpus = _target_cuda_architectures()
    arch_str = ";".join(str(a) for a in architectures)
    if len(architectures) < len(_supported_cuda_architectures()[0]):
        print(f"\nBuilding only for sm_{arch_str}.\n")
    else:
        print(f"\nBuilding for architecture(s) {arch_str}. This can take a "
              "while since more than one is included.\n")

    if not _ask_yes_no(
        "Attempt the CUDA build now? (answer 'n' if you don't have nvcc/a "
        "C++ compiler set up yet - you can rerun `python setup.py` later)"
    ):
        print("Skipped. To build it manually later, set CMAKE_ARGS and pip "
              "install:")
        _print_manual_llama_cpp_command(architectures)
        print("\nOr set MAYA1_USE_GGUF=false in .env to use the "
              "full-precision transformers backend instead (needs 16 GB+ "
              "VRAM, no llama-cpp-python required).")
        return

    env = os.environ.copy()
    env["CMAKE_ARGS"] = f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={arch_str}"
    # Compile the many per-architecture kernel files concurrently instead of
    # one at a time - the other big multiplier on build time alongside
    # trimming the architecture list above.
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 4))
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir",
           "--force-reinstall", "--upgrade", "--verbose", "llama-cpp-python"]
    print(f"$ CMAKE_ARGS=\"{env['CMAKE_ARGS']}\" "
          f"CMAKE_BUILD_PARALLEL_LEVEL={env['CMAKE_BUILD_PARALLEL_LEVEL']} "
          f"{' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("\nBuild failed. Common causes: nvcc not on PATH, no C++ "
              "compiler found, or CUDA toolkit version mismatch with your "
              "driver. Prerequisites again:\n")
        print(_llama_cpp_prereqs_text())
        print("\nOnce prerequisites are installed, rerun `python setup.py`, "
              "or build manually with:")
        _print_manual_llama_cpp_command(architectures)
        print("\nAlternatively, set MAYA1_USE_GGUF=false in .env to use the "
              "full-precision transformers backend instead (no "
              "llama-cpp-python required, but needs 16 GB+ VRAM).")
    else:
        print("\nllama-cpp-python built successfully with CUDA support.")


def _print_manual_llama_cpp_command(architectures: list[int]) -> None:
    arch_str = ";".join(str(a) for a in architectures)
    cpu_count = os.cpu_count() or 4
    system = platform.system()
    if system == "Windows":
        print(
            f"  set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={arch_str}\n"
            f"  set CMAKE_BUILD_PARALLEL_LEVEL={cpu_count}\n"
            "  pip install --no-cache-dir --force-reinstall --upgrade llama-cpp-python"
        )
    else:
        print(
            f"  CMAKE_ARGS=\"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={arch_str}\" \\\n"
            f"  CMAKE_BUILD_PARALLEL_LEVEL={cpu_count} \\\n"
            "  pip install --no-cache-dir --force-reinstall --upgrade llama-cpp-python"
        )


def main() -> None:
    _print_header("Maya1 TTS server - bare-metal setup")
    print("This installs everything needed to run `python server.py` "
          "directly (no Docker). Run it from inside your activated virtual "
          "environment.\n")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.release()}")

    step_env_file()
    step_torch()
    step_requirements()
    step_llama_cpp_python()

    _print_header("Setup finished")
    print("Next steps:\n"
          "  1. (Optional) Edit .env to change the port/model/quantization.\n"
          "  2. Start the server:  python server.py\n"
          "  3. Check it's up:     curl http://localhost:41217/health\n")


if __name__ == "__main__":
    main()
