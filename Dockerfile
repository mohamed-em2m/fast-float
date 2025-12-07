# Base image with CUDA 12.1 for PyTorch
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy everything
COPY . .
ENV IS_IT_RUN_IN_DOCKER=true
# Run the unified setup script
RUN chmod +x setup.sh && ./setup.sh

# Expose port
EXPOSE 8000

# Run the application
CMD [".venv/bin/hypercorn", "main:app", "--bind", "0.0.0.0:8000", "--worker-class", "asyncio", "--log-level", "info", "--access-logfile", "-"]