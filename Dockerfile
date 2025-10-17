FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /workspace

# --- OS deps ---
RUN apt-get update && apt-get install -y \
    git git-lfs aria2 unzip zip rsync nano htop psmisc \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# --- Python base tooling first (helps wheels resolve cleanly) ---
RUN python -m pip install -U pip setuptools wheel typing_extensions

# --- Install Python deps one-by-one with retries so we see which fails ---
# Also keeps the Dockerfile syntax simple (no tricky continuations).
RUN bash -lc '\
  set -e; \
  for PKG in \
    "tensorboard" \
    "matplotlib" \
    "prompt-toolkit" \
    "huggingface_hub" \
    "accelerate" \
    "opencv-python-headless" \
    "gradio==4.45.0" \
  ; do \
    echo "=== Installing $PKG ==="; \
    python -m pip install --prefer-binary "$PKG" || \
    (echo "Retry 1: $PKG" && sleep 5 && python -m pip install --prefer-binary "$PKG") || \
    (echo "Retry 2: $PKG" && sleep 10 && python -m pip install --prefer-binary "$PKG"); \
  done'

EXPOSE 7860

# --- Create directory structure ---
RUN mkdir -p \
    /workspace/models/diffusion_models \
    /workspace/models/text_encoders \
    /workspace/models/vae \
    /workspace/datasets/character_images \
    /workspace/datasets/val \
    /workspace/outputs \
    /workspace/cache \
    /workspace/scripts \
    /workspace/configs \
    /root/.cache/huggingface/accelerate

# --- Copy bootstrap + app ---
COPY wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh
COPY app.py /workspace/app.py
RUN chmod +x /usr/local/bin/wan22_bootstrap.sh

WORKDIR /workspace
# Start command remains in your RunPod template; bootstrap launches Gradio.
