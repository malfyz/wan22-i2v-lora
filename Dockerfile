# Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Always use conda's python/pip
ENV PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off

# Runtime paths used by bootstrap
ENV WORKDIR=/workspace \
    MODELS_DIR=/workspace/models \
    DATASETS_DIR=/workspace/datasets \
    OUTPUTS_DIR=/workspace/outputs \
    CACHE_DIR=/workspace/cache

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

# --- Training deps (NO GRADIO) ---
# Install HF hub with CLI so 'hf' command exists
RUN set -e; \
  retry() { n=0; until [ $n -ge 3 ]; do "$@" && return 0; n=$((n+1)); echo "Retry $n: $*"; sleep $((5*n)); done; return 1; }; \
  PKGS=( \
    "protobuf>=3.20,<5" \
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

# --- Quick import sanity (no Gradio) ---
RUN /opt/conda/bin/python - <<'PY'
mods = ["cv2","accelerate","huggingface_hub","matplotlib","tensorboard","prompt_toolkit","safetensors","tqdm","scipy","numpy","datasets","transformers","peft"]
for m in mods:
    __import__(m)
print("OK imports:", mods)
PY

# --- Copy your entire repo (keeps YOUR wan22_bootstrap._
