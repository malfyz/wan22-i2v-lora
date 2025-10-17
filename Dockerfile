# syntax=docker/dockerfile:1.7

FROM busybox:1.36 AS precheck
# If this COPY fails, your .dockerignore is excluding the file.
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

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs aria2 wget unzip zip rsync nano htop psmisc curl \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    fonts-dejavu-core dos2unix ca-certificates && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# --- Python toolchain ---
RUN python -m pip install -U pip setuptools wheel typing_extensions "protobuf>=3.20,<5"

# --- Lean runtime deps (no musubi here) ---
RUN python -m pip install --prefer-binary --no-cache-dir \
    tensorboard==2.17.1 \
    prompt_toolkit==3.0.48 \
    "huggingface_hub[cli]==0.25.2" \
    accelerate==1.1.1 \
    opencv-python-headless==4.10.0.84 \
    "safetensors>=0.4.4" \
    "tqdm>=4.66.5" \
    "numpy>=1.26.4" \
    "transformers>=4.44.2" \
    "peft>=0.11.1" \
 && conda clean -afy || true \
 && rm -rf /opt/conda/pkgs/* /root/.cache/pip || true

# --- Your repo ---
COPY . /workspace

# --- Bake bootstrap into a stable path (if present) ---
COPY --from=precheck /wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh
RUN dos2unix /usr/local/bin/wan22_bootstrap.sh || true && chmod +x /usr/local/bin/wan22_bootstrap.sh
RUN if [ -f /workspace/wan22_bootstrap.sh ]; then dos2unix /workspace/wan22_bootstrap.sh || true; chmod +x /workspace/wan22_bootstrap.sh || true; fi

# --- Ensure dirs ---
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

# --- Self-healing startup wrapper (HEADLESS, no Gradio) ---
RUN printf '%s\n' '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'echo "[STARTUP] Python: $(python -V 2>&1)"' \
  'BOOT=""' \
  'for p in /workspace/wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh; do' \
  '  if [ -f "$p" ]; then BOOT="$p"; break; fi' \
  'done' \
  'if [ -z "$BOOT" ]; then' \
  '  echo "[STARTUP] bootstrap missing — fetching from repo…"' \
  '  mkdir -p /workspace' \
  '  RAW_URL="https://raw.githubusercontent.com/malfyz/wan22-i2v-lora/main/wan22_bootstrap.sh"' \
  '  curl -fsSL "$RAW_URL" -o /workspace/wan22_bootstrap.sh' \
  '  BOOT=/workspace/wan22_bootstrap.sh' \
  'fi' \
  'sed -i "s/\r$//" "$BOOT" || true' \
  'chmod +x "$BOOT" || true' \
  'echo "[STARTUP] running $BOOT (headless)"' \
  'bash "$BOOT" || echo "[STARTUP] bootstrap exited non-zero (continuing so you can SSH)"' \
  'echo "[STARTUP] Pod alive for SSH…"; tail -f /dev/null' \
  > /usr/local/bin/startup.sh && chmod +x /usr/local/bin/startup.sh

WORKDIR /workspace
# IMPORTANT: use CMD (so RunPod can override). Do NOT use ENTRYPOINT.
CMD ["/bin/bash","-lc","/usr/local/bin/startup.sh"]
