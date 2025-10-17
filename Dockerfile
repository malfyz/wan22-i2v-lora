FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /workspace

# --- OS deps ---
# 🔧 Add bash explicitly (some runtime images omit it). Add fonts for matplotlib.
RUN apt-get update && apt-get install -y \
    bash git git-lfs aria2 unzip zip rsync nano htop psmisc \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# 🔧 Make bash the default shell for subsequent RUN steps
SHELL ["/bin/bash", "-lc"]

# --- Python base tooling first (helps wheels resolve cleanly) ---
RUN python -m pip install -U pip setuptools wheel typing_extensions

# 🔧 Pre-pin protobuf to avoid occasional tensorboard resolver issues
RUN python -m pip install "protobuf<5"

# --- Install Python deps one-by-one with retries so we see which fails ---
# Also keeps the Dockerfile syntax simple (no tricky continuations).
RUN set -e; \
  retry() { n=0; until [ $n -ge 3 ]; do "$@" && break; n=$((n+1)); echo "Retry $n: $*"; sleep $((5*n)); done; }; \
  for PKG in \
    "tensorboard" \
    "matplotlib" \
    "prompt-toolkit" \
    "huggingface_hub" \
    "accelerate" \
    "opencv-python-headless==4.10.0.84" \
    "gradio==4.45.0" \
  ; do \
    echo "=== Installing $PKG ==="; \
    retry python -m pip install --prefer-binary --no-cache-dir "$PKG"; \
  done

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
