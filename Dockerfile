# Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prefer conda python at runtime & pip defaults
ENV PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_PROGRESS_BAR=off \
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

# --- Keep toolchain current (use conda's python explicitly) ---
RUN /opt/conda/bin/python -m pip install -U pip setuptools wheel typing_extensions
# TensorBoard sometimes wants protobuf<5
RUN /opt/conda/bin/python -m pip install "protobuf<5"

# --- Python deps (install one-by-one with retries & clear logs) ---
# NOTE: prompt_toolkit has an underscore (not a hyphen)
RUN set -e; \
  retry() { n=0; until [ $n -ge 3 ]; do "$@" && return 0; n=$((n+1)); echo "Retry $n: $*"; sleep $((5*n)); done; return 1; }; \
  PKGS=( \
    "tensorboard" \
    "matplotlib" \
    "prompt_toolkit" \
    "huggingface_hub" \
    "accelerate" \
    "opencv-python-headless==4.10.0.84" \
    "gradio==4.45.0" \
  ); \
  for PKG in "${PKGS[@]}"; do \
    echo "=== Installing $PKG ==="; \
    retry /opt/conda/bin/python -m pip install --prefer-binary --no-cache-dir "$PKG" || { echo "FAILED: $PKG"; exit 1; }; \
  done

# --- Validate at build-time (fail now if imports are broken) ---
RUN /opt/conda/bin/python - <<'PY'
mods = ["gradio","cv2","accelerate","huggingface_hub","matplotlib","tensorboard","prompt_toolkit"]
for m in mods:
    __import__(m)
print("OK imports:", mods)
PY

# --- Copy your project ---
COPY . /workspace

# Also copy bootstrap to a stable path and mark executable
RUN cp /workspace/wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh && \
    chmod +x /usr/local/bin/wan22_bootstrap.sh

# Expose Gradio port
EXPOSE 7860

# Start: list /workspace then launch bootstrap
ENTRYPOINT ["/bin/bash","-lc","echo '[ENTRYPOINT] /workspace listing:'; ls -la /workspace || true; /usr/local/bin/wan22_bootstrap.sh"]
