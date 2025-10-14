FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Minimal OS deps
RUN apt-get update && apt-get install -y \
    git git-lfs aria2 unzip zip rsync nano htop psmisc ca-certificates \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# Python deps (Torch already present in base image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorboard matplotlib prompt-toolkit "huggingface_hub==0.25.*"

# Dirs used at runtime
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

# Accelerate default (single GPU, bf16)
RUN printf '%s\n' \
  'compute_environment: LOCAL_MACHINE' \
  'distributed_type: NO' \
  'gpu_ids: "0"' \
  'mixed_precision: bf16' \
  'num_machines: 1' \
  'num_processes: 1' \
  'main_training_function: main' \
  > /root/.cache/huggingface/accelerate/default_config.yaml

# Musubi-Tuner (WAN 2.2 LoRA trainer)
RUN git clone https://github.com/kohya-ss/musubi-tuner.git /workspace/musubi-tuner && \
    pip install --no-cache-dir -e /workspace/musubi-tuner

# Copy bootstrap script into the image
COPY wan22_bootstrap.sh /usr/local/bin/wan22_bootstrap.sh
RUN chmod +x /usr/local/bin/wan22_bootstrap.sh

WORKDIR /workspace
# No ENTRYPOINT; RunPod will invoke the start command you set in the template.
