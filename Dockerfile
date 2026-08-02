# syntax=docker/dockerfile:1
#
# Caldera Engine — GPU-first image (falls back to CPU via build args).
#
# The container needs only the CUDA *runtime* (bundled in the cu121 torch wheel);
# the NVIDIA driver is supplied by the host via the NVIDIA Container Toolkit +
# `--gpus all`. See docs/Caldera Engine Docker Deployment — Plan.md.
#
# GPU build (default):   docker compose build
# CPU build:             docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu

ARG PYTHON_BASE=python:3.11-slim

# ---- builder: install deps into a venv ---------------------------------------
FROM ${PYTHON_BASE} AS builder
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements.txt .
# Install (CUDA or CPU) torch first so the requirements install sees it satisfied
# and doesn't pull a conflicting CPU build over it; then the rest; then the spaCy model.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchaudio --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_lg

# ---- runtime -----------------------------------------------------------------
FROM ${PYTHON_BASE} AS runtime
# ffmpeg is a hard runtime dep (production_mixer / audio_generation shell out to it);
# curl is for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TTS_HOME=/app/.cache/tts
WORKDIR /app
COPY . .
# Non-root. State dirs are created here so named volumes mounted over them inherit
# this ownership on first creation (Docker copies the mountpoint's perms).
RUN useradd -m caldera \
    && mkdir -p /app/data /app/scratch /app/.cache \
    && chown -R caldera:caldera /app
USER caldera
EXPOSE 8082
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8082/ || exit 1
CMD ["python", "-m", "src.gui_server"]
