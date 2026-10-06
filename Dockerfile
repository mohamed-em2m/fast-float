# Base image with CUDA 12.1 for PyTorch
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# uv binary (official image) — setup.sh uses it instead of pip
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy everything
COPY . .
ENV IS_IT_RUN_IN_DOCKER=true
ENV PATH="/root/.local/bin:${PATH}"
# Run the unified setup script (it lives in scripts/)
RUN chmod +x scripts/setup.sh scripts/download_checkpoints.sh && ./scripts/setup.sh

# Expose port
EXPOSE 8000

# Run the application
CMD [".venv/bin/hypercorn", "main:app", "--bind", "0.0.0.0:8000", "--worker-class", "asyncio", "--log-level", "info", "--access-logfile", "-"]