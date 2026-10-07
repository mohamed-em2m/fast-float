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
RUN chmod +x ./setup.sh
RUN ./setup.sh

# Expose port (Cloud Run uses $PORT env variable)
EXPOSE 8080
ENV PORT=8080

# Use shell form to allow environment variable expansion
CMD .venv/bin/hypercorn main:app --bind 0.0.0.0:${PORT} --log-level info --startup-timeout 600 --workers 1