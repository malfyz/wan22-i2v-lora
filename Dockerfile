# Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_PROGRESS_BAR=off \
    PATH=/opt/conda/bin:$PATH \
    WORKDIR=/workspace \
    MODELS_DIR=/workspace/models \
    DATASETS_DIR=/workspace/datasets \
    OUTPUTS_DIR=/workspace/outputs \
    CACHE_DIR=/workspace/cache

WORKDIR /workspace
SHELL ["/bin/bash","-lc"]

# --- OS deps (no UI stuff) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs aria2 wget unzip zip rsync nano htop psmisc curl \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    fonts-dejavu-core dos2unix ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# --- Python toolchain ---
RUN python -m pip install -U pip setuptools wheel typing_extensions "protobuf>=3.20,<5"

# --- Core training/runtime deps (NO gradio) ---
RUN set -e; \
  retry() { n=0; until [ $n -ge 3 ]; do "$@" && return 0; n=$((n+1)); echo "Retry $n: $*"; sleep $((5*n)); done; return 1; }; \
  PKGS=( \
    "tensorboard==2.17.1" \
    "matplotlib==3.9.2" \
    "prompt_toolkit==3.0.48" \
    "huggingface_hub[cli]==0.25.2" \
    "accelerate==1.1.1" \
    "opencv-python-headless==4.10.0.84" \
    "safetensors>=0.4.4" \
    "tqdm>=4.66.5" \
    "scipy>=1.11.4" \
    "numpy>=1.26.4" \
    "datasets>=2.21.0" \
    "transformers>=4.44.2" \
    "peft>=0.11.1" \
  ); \
  for PKG in "${PKGS[@]}"; do \
    echo "=== Installing $PKG ==="; \
    retry python -m pip install --prefer-binary --no-cache-dir "$PKG"; \
  done

# --- Musubi-Tuner (WAN 2.2 LoRA trainer) ---
# install from GitHub in editable mode so scripts are available
RUN cd /opt && git clone https://github.com/kohya-ss/musubi-tuner.git && \
    cd /opt/musubi-tuner && python -m pip install -e .

# --- Copy repo into /workspace (ensures your scripts/configs are present) ---
COPY . /workspace

# --- Install bootstrap to a stable path and also keep it in /workspace ---
# (If your repo has wan22_bootstrap.sh, this COPY ensures it’s inside the image)
COPY wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh
RUN test -f /usr/local/bin/wan22_bootstrap.sh || (echo "FATAL: wan22_bootstrap.sh missing from build context" && exit 1)
RUN dos2unix /usr/local/bin/wan22_bootstrap.sh || true
RUN chmod +x /usr/local/bin/wan22_bootstrap.sh

# If it also exists in /workspace, normalize it; otherwise we’ll restore at start.
RUN if [ -f /workspace/wan22_bootstrap.sh ]; then dos2unix /workspace/wan22_bootstrap.sh || true; chmod +x /workspace/wan22_bootstrap.sh || true; fi

# --- Ensure expected dirs exist ---
RUN mkdir -p /workspace/models/diffusion_models \
             /workspace/models/text_encoders \
             /workspace/models/vae \
             /workspace/datasets/character_images \
             /workspace/datasets/val \
             /workspace/outputs \
             /workspace/cache \
             /workspace/scripts \
             /workspace/configs \
             /root/.cache/huggingface/accelerate

# --- Startup wrapper ---
# 1) restore /workspace/wan22_bootstrap.sh from /usr/local/bin if missing
# 2) run bootstrap (models/datasets + training scripts creation)
# 3) keep the container alive so RunPod SSH works even if bootstrap exits
RUN printf '%s\n' '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'echo "[STARTUP] Ensuring bootstrap at /workspace/wan22_bootstrap.sh..."' \
  'if [ ! -f /workspace/wan22_bootstrap.sh ]; then' \
  '  cp -f /usr/local/bin/wan22_bootstrap.sh /workspace/wan22_bootstrap.sh' \
  '  chmod +x /workspace/wan22_bootstrap.sh || true' \
  'fi' \
  'sed -i "s/\r$//" /workspace/wan22_bootstrap.sh || true' \
  'echo "[STARTUP] Running bootstrap..."' \
  '/workspace/wan22_bootstrap.sh || echo "[STARTUP] Bootstrap exited non-zero (continuing so you can SSH)"' \
  'echo "[STARTUP] Pod alive for SSH…"; tail -f /dev/null' \
  > /usr/local/bin/startup.sh && chmod +x /usr/local/bin/startup.sh

WORKDIR /workspace
ENTRYPOINT ["/bin/bash","-lc","/usr/local/bin/startup.sh"]
