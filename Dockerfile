# syntax=docker/dockerfile:1.7

FROM busybox:1.36 AS precheck
COPY wan22_bootstrap.sh /wan22_bootstrap.sh
RUN test -f /wan22_bootstrap.sh

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

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs aria2 wget unzip zip rsync nano htop psmisc curl \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    fonts-dejavu-core dos2unix ca-certificates && \
    rm -rf /var/lib/apt/lists/* && git lfs install

RUN python -m pip install -U pip setuptools wheel typing_extensions "protobuf>=3.20,<5"

# Lean deps (no scipy/matplotlib/datasets; NO musubi here)
RUN set -e; \
  PKGS=( \
    "tensorboard==2.17.1" \
    "prompt_toolkit==3.0.48" \
    "huggingface_hub[cli]==0.25.2" \
    "accelerate==1.1.1" \
    "opencv-python-headless==4.10.0.84" \
    "safetensors>=0.4.4" \
    "tqdm>=4.66.5" \
    "numpy>=1.26.4" \
    "transformers>=4.44.2" \
    "peft>=0.11.1" \
  ); \
  python -m pip install --prefer-binary --no-cache-dir "${PKGS[@]}" && \
  conda clean -afy || true && rm -rf /opt/conda/pkgs/* /root/.cache/pip || true

# Put your source tree in /workspace
COPY . /workspace

# Bake bootstrap into a stable path; also restore into /workspace at start
COPY --from=precheck /wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh
RUN dos2unix /usr/local/bin/wan22_bootstrap.sh || true && chmod +x /usr/local/bin/wan22_bootstrap.sh
RUN if [ -f /workspace/wan22_bootstrap.sh ]; then dos2unix /workspace/wan22_bootstrap.sh || true; chmod +x /workspace/wan22_bootstrap.sh || true; fi

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

RUN printf '%s\n' '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'if [ ! -f /workspace/wan22_bootstrap.sh ]; then cp -f /usr/local/bin/wan22_bootstrap.sh /workspace/wan22_bootstrap.sh; fi' \
  'sed -i "s/\r$//" /workspace/wan22_bootstrap.sh || true' \
  'chmod +x /workspace/wan22_bootstrap.sh || true' \
  'echo "[STARTUP] Running bootstrap..."' \
  '/workspace/wan22_bootstrap.sh || echo "[STARTUP] Bootstrap exited non-zero (continuing so you can SSH)"' \
  'echo "[STARTUP] Pod alive for SSH…"; tail -f /dev/null' \
  > /usr/local/bin/startup.sh && chmod +x /usr/local/bin/startup.sh

WORKDIR /workspace
ENTRYPOINT ["/bin/bash","-lc","/usr/local/bin/startup.sh"]
