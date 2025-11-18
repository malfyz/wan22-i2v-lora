#!/usr/bin/env bash
set -euo pipefail
echo "[BOOTSTRAP] WAN 2.1 T2V LoRA start $(date -Iseconds)"

WORKDIR="/workspace"
MODELS="$WORKDIR/models"
DATA="$WORKDIR/datasets"
CACHE="$WORKDIR/cache"
CONF="$WORKDIR/configs"
OUT="$WORKDIR/outputs"
SCRIPTS="$WORKDIR/scripts"
LOGS="$WORKDIR/logs"

mkdir -p "$MODELS"/{dit,vae,t5} "$DATA" "$CACHE/t2v" "$CONF" "$OUT" "$SCRIPTS" "$LOGS"

# Install musubi-tuner
pip install -U pip wheel setuptools
pip install "huggingface_hub[cli]==0.25.2"
pip install git+https://github.com/kohya-ss/musubi-tuner.git
pip install hf_transfer==0.1.8
export HF_HUB_ENABLE_HF_TRANSFER=1

HF=huggingface-cli

# Download models
echo "[MODELS] Downloading WAN 2.1 T2V models..."

$HF download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors \
  --local-dir "$MODELS/dit" --local-dir-use-symlinks=False

$HF download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/vae/wan_2.1_vae.safetensors \
  --local-dir "$MODELS/vae" --local-dir-use-symlinks=False

$HF download Wan-AI/Wan2.1-I2V-14B-720P \
  models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir "$MODELS/t5" --local-dir-use-symlinks=False

echo "[MODELS] Done."

# Dataset config
cat > "$CONF/dataset_t2v.toml" <<TOML
[general]
resolution = [960, 544]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
caption_extension = ".txt"

[[datasets]]
image_directory = "/workspace/datasets"
cache_directory = "/workspace/cache/t2v"
num_repeats = 1
TOML

# Export memory-friendly flags
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Script: train
cat > "$SCRIPTS/train_t2v.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

CONF="/workspace/configs/dataset_t2v.toml"
M="/workspace/models"
LOG="/workspace/logs/train.log"

echo "[CACHE] Latents..."
python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"

echo "[CACHE] Text Encoder outputs..."
python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/t5/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda --batch_size 4 | tee -a "$LOG"

echo "[TRAIN] Launching..."
accelerate launch -m musubi_tuner.wan_train_network \
  --task t2v-14B \
  --dataset_config "$CONF" \
  --dit "$M/dit/wan2.1_t2v_14B_fp16.safetensors" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --t5 "$M/t5/models_t5_umt5-xxl-enc-bf16.pth" \
  --network_module networks.lora_wan \
  --network_dim 32 \
  --network_alpha 32 \
  --learning_rate 1e-4 \
  --optimizer_type adamw8bit \
  --gradient_checkpointing \
  --mixed_precision fp16 \
  --sdpa \
  --max_train_epochs 20 \
  --save_every_n_epochs 5 \
  --output_dir "/workspace/outputs/t2v" \
  --output_name "wan21_t2v_lora" | tee -a "$LOG"

echo "[TRAIN] Done."
BASH

chmod +x "$SCRIPTS/train_t2v.sh"

echo "[BOOTSTRAP] Completed."
