# Base image. CUDA 12.6, matching the pinned wheels (torch cu126 +
# onnxruntime-gpu 1.23 both dlopen CUDA 12 sonames). The NVIDIA *driver* comes from the host, so
# the container tag mainly has to agree with what the venv dlopens.
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependency manifests + uv first, so a code-only change does not re-extract
#    the ~7 GiB torch/cu126 wheels in every rebuild: scripts/setup.sh reuses
#    the venv created here and its `uv sync --inexact` is then a no-op.
COPY pyproject.toml uv.lock .python-version ./
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && export PATH="${HOME}/.local/bin:${PATH}" \
    && uv sync --inexact

# 2) The rest of the tree. .dockerignore keeps git metadata, node_modules, the
#    .venv, results/ and the unused cloudflared .deb out of the build context.
COPY . .
ENV IS_IT_RUN_IN_DOCKER=true

# 3) Unified setup: best-effort system packages, Node 22 for the frontend
#    (only needed if the image is later extended; harmless in the API image),
#    venv reuse, the ONNX/TensorRT layer, and checkpoints (skipped when the
#    build context already carries them).
RUN chmod +x ./scripts/*.sh && ./scripts/setup.sh

# Cloud Run injects $PORT; plain `docker run` uses 8080.
EXPOSE 8080
ENV PORT=8080

# Entrypoint is main.py's __main__ block, not the `hypercorn` CLI:
# * the old CMD passed `--startup-timeout 600`, which hypercorn does not have
#   (that is uvicorn) — hypercorn aborted with an unrecognized-argument error;
# * main.py maps env -> config itself: it reads $PORT and raises hypercorn's
#   startup_timeout to 600 s (the CLI leaves the 60 s default, which a cold
#   model load can exceed).
CMD ["sh", "-c", ".venv/bin/python fast_float/main.py"]
