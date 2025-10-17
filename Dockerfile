# Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /workspace

# --- OS deps (bash + libs for opencv/matplotlib/ffmpeg) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git git-lfs aria2 unzip zip rsync nano htop psmisc \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Use bash for RUN steps
SHELL ["/bin/bash", "-lc"]

# --- Keep pip toolchain current ---
RUN python -m pip install -U pip setuptools wheel typing_extensions

# --- Pre-pin to avoid occasional tensorboard/protobuf resolver issues ---
RUN python -m pip install "protobuf<5"

# --- Install Python deps, with retries (before copying source for better cache) ---
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

# --- Copy your entire repo into /workspace (ensures app.py is present) ---
COPY . /workspace

# Make bootstrap executable
RUN chmod +x /workspace/wan22_bootstrap.sh

# Expose Gradio port
EXPOSE 7860

# Default working dir
WORKDIR /workspace

# Launch bootstrap (which runs the Gradio app)
ENTRYPOINT ["/bin/bash","-lc","/workspace/wan22_bootstrap.sh"]
