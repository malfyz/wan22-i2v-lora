# Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Always use conda's python/pip (the base image ships with it)
ENV PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /workspace

# --- OS deps (bash + libs for opencv/matplotlib/ffmpeg/fonts) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git git-lfs aria2 unzip zip rsync nano htop psmisc \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Use bash for RUN steps
SHELL ["/bin/bash", "-lc"]

# --- Keep toolchain current (on conda's python) ---
RUN /opt/conda/bin/python -m pip install -U pip setuptools wheel typing_extensions

# --- Pre-pin protobuf for tensorboard compatibility ---
RUN /opt/conda/bin/python -m pip install "protobuf>=3.20,<5"

# --- Known-good pins (compatible with PyTorch 2.5 + Python 3.11) ---
# If any of these still fail in your environment, the loop below will print
# exactly which one failed before exiting.
RUN set -e; \
  retry() { n=0; until [ $n -ge 3 ]; do "$@" && return 0; n=$((n+1)); echo "Retry $n: $*"; sleep $((5*n)); done; return 1; }; \
  PKGS=( \
    "tensorboard==2.17.1" \
    "matplotlib==3.9.2" \
    "prompt_toolkit==3.0.48" \
    "huggingface_hub==0.25.2" \
    "accelerate==1.1.1" \
    "opencv-python-headless==4.10.0.84" \
    "gradio==4.45.0" \
  ); \
  for PKG in "${PKGS[@]}"; do \
    echo "=== Installing $PKG ==="; \
    if ! retry /opt/conda/bin/python -m pip install --prefer-binary --no-cache-dir "$PKG" 2>&1 | tee "/tmp/pip_${PKG//[^A-Za-z0-9._-]/_}.log"; then \
      echo "----- PIP LOG (tail) for $PKG -----"; \
      tail -n 200 "/tmp/pip_${PKG//[^A-Za-z0-9._-]/_}.log" || true; \
      echo "FAILED: $PKG"; exit 1; \
    fi; \
  done

# --- Validate at build-time (fail now if imports are broken) ---
RUN /opt/conda/bin/python - <<'PY'
mods = ["gradio","cv2","accelerate","huggingface_hub","matplotlib","tensorboard","prompt_toolkit"]
for m in mods:
    __import__(m)
print("OK imports:", mods)
PY

# --- Copy your project into the image ---
COPY . /workspace

# Place bootstrap in a guaranteed path and make it executable
RUN cp /workspace/wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh && \
    chmod +x /usr/local/bin/wan22_bootstrap.sh

# Expose Gradio port
EXPOSE 7860

# On container start: show /workspace contents, then run bootstrap
ENTRYPOINT ["/bin/bash","-lc","echo '[ENTRYPOINT] /workspace listing:'; ls -la /workspace || true; /usr/local/bin/wan22_bootstrap.sh"]
