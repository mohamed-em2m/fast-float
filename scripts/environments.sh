# Manual install equivalent of scripts/setup.sh (run from the repo root).
# Requires `uv` on PATH. Python version comes from .python-version,
# CUDA PyTorch pins and all deps come from pyproject.toml + uv.lock.
uv sync --frozen
