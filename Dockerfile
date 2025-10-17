# Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    WORKDIR=/workspace \
    MODELS_DIR=/workspace/models \
    DATASETS_DIR=/workspace/datasets \
    OUTPUTS_DIR=/workspace/outputs \
    CACHE_DIR=/workspace/cache

WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]

# ---- OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git git-lfs aria2 unzip zip rsync nano htop psmisc \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    fonts-dejavu-core dos2unix \
 && rm -rf /var/lib/apt/lists/*

# ---- Python toolchain
RUN /opt/conda/bin/python -m pip install -U pip setuptools wheel typing_extensions
RUN /opt/conda/bin/python -m pip install "protobuf>=3.20,<5"

# ---- Training deps (NO gradio)
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
    retry /opt/conda/bin/python -m pip install --prefer-binary --no-cache-dir "$PKG"; \
  done

# ---- COPY THE BOOTSTRAP FIRST to a stable location; fail fast if missing
# If your file is named differently, FIX THE NAME BELOW to match the repo exactly.
COPY wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh
RUN test -f /usr/local/bin/wan22_bootstrap.sh || (echo "FATAL: wan22_bootstrap.sh not found in build context"; exit 1)
RUN dos2unix /usr/local/bin/wan22_bootstrap.sh || true
RUN chmod +x /usr/local/bin/wan22_bootstrap.sh

# ---- Now copy the entire repo to /workspace (for your code/configs)
COPY . /workspace

# Ensure dirs your script uses exist
RUN mkdir -p /workspace/models/diffusion_models \
             /workspace/models/text_encoders \
             /workspace/models/vae \
             /workspace/datasets \
             /workspace/outputs \
             /workspace/cache

# ---- Startup wrapper: run bootstrap from the STABLE PATH, then keep alive
RUN printf '%s\n' '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'echo "[STARTUP] Listing /workspace:"' \
    'ls -la /workspace || true' \
    'echo "[STARTUP] Checking bootstrap at /usr/local/bin/wan22_bootstrap.sh…"' \
    'if [ ! -x /usr/local/bin/wan22_bootstrap.sh ]; then echo "[STARTUP] FATAL: bootstrap missing"; exit 1; fi' \
    'echo "[STARTUP] Running bootstrap…"' \
    '/usr/local/bin/wan22_bootstrap.sh || echo "[STARTUP] Bootstrap exited non-zero (continuing so SSH works)."' \
    'echo "[STARTUP] Container will stay up for SSH…"' \
    'while true; do sleep 3600; done' \
    > /usr/local/bin/startup.sh && chmod +x /usr/local/bin/startup.sh

ENTRYPOINT ["/bin/bash","-lc","/usr/local/bin/startup.sh"]
