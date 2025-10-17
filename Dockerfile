# Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Use conda's python/pip everywhere
ENV PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off

WORKDIR /workspace

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git git-lfs aria2 unzip zip rsync nano htop psmisc \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Use bash for RUN steps
SHELL ["/bin/bash", "-lc"]

# --- Python toolchain (conda python) ---
RUN /opt/conda/bin/python -m pip install -U pip setuptools wheel typing_extensions

# TensorBoard sometimes prefers protobuf<5
RUN /opt/conda/bin/python -m pip install "protobuf>=3.20,<5"

# --- Python deps (NO GRADIO) ---
# We install one-by-one with retries so failures are obvious.
RUN set -e; \
  retry() { n=0; until [ $n -ge 3 ]; do "$@" && return 0; n=$((n+1)); echo "Retry $n: $*"; sleep $((5*n)); done; return 1; }; \
  PKGS=( \
    "tensorboard==2.17.1" \
    "matplotlib==3.9.2" \
    "prompt_toolkit==3.0.48" \
    "huggingface_hub==0.25.2" \
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
    if ! retry /opt/conda/bin/python -m pip install --prefer-binary --no-cache-dir "$PKG" 2>&1 | tee "/tmp/pip_${PKG//[^A-Za-z0-9._-]/_}.log"; then \
      echo "----- PIP LOG (tail) for $PKG -----"; \
      tail -n 200 "/tmp/pip_${PKG//[^A-Za-z0-9._-]/_}.log" || true; \
      echo "FAILED: $PKG"; exit 1; \
    fi; \
  done

# --- Quick import sanity (no Gradio) ---
RUN /opt/conda/bin/python - <<'PY'
mods = ["cv2","accelerate","huggingface_hub","matplotlib","tensorboard","prompt_toolkit","safetensors","tqdm","scipy","numpy","datasets","transformers","peft"]
for m in mods:
    __import__(m)
print("OK imports:", mods)
PY

# --- Copy your project into the image ---
COPY . /workspace

# --- Create a tiny keepalive so the container never exits (SSH can attach) ---
RUN printf '%s\n' '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'echo "[KEEPALIVE] Container up. $(date)";' \
    'echo "[KEEPALIVE] Python: $(python --version)";' \
    'echo "[KEEPALIVE] Listing /workspace:";' \
    'ls -la /workspace || true' \
    'echo "[KEEPALIVE] Sleeping..."' \
    'while true; do sleep 3600; done' \
    > /usr/local/bin/keepalive.sh && chmod +x /usr/local/bin/keepalive.sh

# Default working dir
WORKDIR /workspace

# Keep container running for SSH
ENTRYPOINT ["/bin/bash","-lc","/usr/local/bin/keepalive.sh"]
