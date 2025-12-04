# Base image with CUDA 11.8 for PyTorch
FROM anibali/pytorch:2.0.1-cuda11.8
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Create and activate a virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN uv pip install --system --no-cache-dir -r requirements.txt hypercorn==0.18.0 soundfile==0.13.1 && uv pip check
# Upgrade pip
RUN pip install --upgrade pip
# Install PyTorch and its ecosystem
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface-hub
# Copy the rest of the application code
COPY . .
# Download model checkpoints
RUN chmod +x download_checkpoints.sh
RUN ./download_checkpoints.sh
# Create a directory for results if it doesn't exist
RUN mkdir -p results
# Expose the port the app runs on
EXPOSE 8000
# Set the command to run the application
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8080", "--worker-class", "asyncio", "--log-level", "info", "--access-logfile", "-"]
