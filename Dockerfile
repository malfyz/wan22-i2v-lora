FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# --- OS dependencies for OpenCV + media ---
RUN apt-get update && apt-get install -y \
    git git-lfs aria2 unzip zip rsync nano htop psmisc \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
# --- Python dependencies ---
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorboard matplotlib prompt-toolkit huggingface_hub accelerate opencv-python-headless gradio==4.45.0


EXPOSE 7860

# --- Create directory structure ---
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

# --- Copy bootstrap script ---
COPY wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh
RUN chmod +x /usr/local/bin/wan22_bootstrap.sh
COPY app.py /workspace/app.py

WORKDIR /workspace
# No ENTRYPOINT — RunPod will call the start command

